"""DETR-style transformer encoder/decoder for the ACT preset.

PORTED, not imported, from ``third_party/act/detr/models/transformer.py`` (DETR,
Copyright (c) Facebook, Inc. — Apache 2.0), which is itself a copy of
``torch.nn.Transformer`` with three modifications DETR needs and ACT relies on:

1. **Positional encodings are added at every attention layer**, inside the attention
   (``with_pos_embed`` on q/k only, never on v), rather than once at the input. This is
   why the layers cannot be replaced with ``nn.TransformerEncoder`` — the pos-embed
   injection point differs and the learned weights would not transfer.
2. The encoder's trailing LayerNorm is removed (``encoder_norm`` is None unless
   ``normalize_before``).
3. The decoder returns a STACK of all layers' activations (``return_intermediate``),
   which is DETR's auxiliary-loss hook. ACT builds with ``return_intermediate_dec=True``
   and then takes ``[0]`` — the FIRST decoder layer. See :class:`ACTTransformer.forward`.

Everything numerically load-bearing is byte-equivalent to upstream. What changed:

* ``IPython.embed`` debug hooks removed.
* The ``len(src.shape) == 3`` branch of upstream ``Transformer.forward`` (the
  no-backbone, state-only CNNMLP-ish path) is dropped: FUJI's ACT preset always has an
  image, so that branch was dead and its silent fallthrough (no ``latent_input``, no
  ``proprio_input``) would be a wrong-model bug rather than a useful degradation.
* Upstream stacks exactly two extra tokens (``latent``, ``proprio``). Here the extra
  tokens are a LIST, because FUJI's observation has a third conditioning modality —
  force — that ALOHA has no sensor for. See :meth:`ACTTransformer.forward`.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """Self-attention + FFN, with the positional embedding added to q/k only."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    """A stack of :class:`TransformerEncoderLayer`, threading ``pos`` into every layer."""

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Module):
    """Self-attention over queries, cross-attention into memory, FFN."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):
    """A stack of :class:`TransformerDecoderLayer`, optionally returning every layer."""

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None,
                 return_intermediate: bool = False) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None, stop_after: int | None = None):
        """``stop_after=n`` runs only the first ``n`` layers.

        The stack is sequential, so ``intermediate[i]`` depends on layers ``0..i`` and on
        NOTHING deeper: truncating to ``n`` layers leaves entries ``0..n-1`` bit-identical to
        a full run. ACT reads ``hs[0]`` only (``dp_act.py``), so ``stop_after=1`` is a free
        1.61x — the deeper layers are computed and thrown away otherwise. Kept as an argument
        rather than as construction-time depth so the module still holds (and loads) all
        ``dec_layers`` layers' weights: the checkpoint's ``state_dict`` is unchanged, and a
        future readout that wants a deeper index just asks for it.
        """
        n = len(self.layers) if stop_after is None else max(1, min(int(stop_after), len(self.layers)))
        output = tgt
        intermediate = []
        for layer in self.layers[:n]:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class ACTTransformer(nn.Module):
    """The ACT encoder/decoder pair over [conditioning tokens | flattened visual grid].

    Upstream ``Transformer.forward`` hardcodes two extra conditioning tokens stacked in
    front of the visual tokens (``torch.stack([latent_input, proprio_input])``). FUJI's
    observation has THREE modalities — latent z, proprioception, and a force window that
    ALOHA has no sensor for — so ``extra_tokens`` is a list and ``additional_pos_embed``
    is sized to match. With ``extra_tokens=[latent, proprio]`` this is byte-identical to
    upstream; passing a third appends one token and one learned position, leaving the
    visual token positions (and hence the pretrained-backbone alignment) untouched.
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 7, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = False,
                 return_intermediate_dec: bool = False) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, query_embed: Tensor, pos_embed: Tensor,
                extra_tokens: list[Tensor], additional_pos_embed: Tensor,
                decoder_layers: int | None = None) -> Tensor:
        """
        Parameters
        ----------
        src : ``(B, C, H, W)`` projected visual features (already at ``d_model`` channels).
            Multiple cameras/frames are folded into ``W`` by the caller, exactly as
            upstream does (``torch.cat(all_cam_features, axis=3)``).
        query_embed : ``(num_queries, d_model)`` learned action queries.
        pos_embed : ``(1, d_model, H, W)`` visual position embedding.
        extra_tokens : list of ``(B, d_model)`` conditioning tokens, prepended to the
            sequence in the given order. Must have exactly ``additional_pos_embed``'s
            length — asserted, because a silent length mismatch would shift every
            token's learned position by one.
        additional_pos_embed : ``(len(extra_tokens), d_model)`` learned positions for them.
        decoder_layers : run only the first N decoder layers; see
            :meth:`TransformerDecoder.forward`. ``None`` runs the whole stack.

        Returns
        -------
        ``(n_layers_run, B, num_queries, d_model)`` decoder activations.
        """
        if src.dim() != 4:
            raise ValueError(f"ACTTransformer expects (B, C, H, W) visual features, got {tuple(src.shape)}")
        if len(extra_tokens) != additional_pos_embed.shape[0]:
            raise ValueError(
                f"got {len(extra_tokens)} conditioning tokens but additional_pos_embed has "
                f"{additional_pos_embed.shape[0]} rows; a mismatch would shift every token's "
                f"learned position")
        bs = src.shape[0]
        # (B, C, H, W) -> (HW, B, C); MultiheadAttention here is seq-first (no batch_first).
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # (n_extra, B, C)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)
        src = torch.cat([torch.stack(extra_tokens, axis=0), src], axis=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed,
                          stop_after=decoder_layers)
        return hs.transpose(1, 2)

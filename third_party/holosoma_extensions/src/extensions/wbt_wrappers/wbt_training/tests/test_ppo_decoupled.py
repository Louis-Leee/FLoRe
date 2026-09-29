"""Unit tests for the FALCON-style dual-agent PPO (PPODecoupled).

We avoid a full IsaacSim env by exercising the pure/algorithmic pieces on a bare
instance built with __new__ (bypassing __init__) and hand-setting the few attributes
each method reads. The load-bearing invariants tested here:
  * action scatter/gather permutation round-trips (no joint-crossing bug),
  * the fused dual-head ONNX wrapper emits a 29-D action in ENV JOINT ORDER,
  * per-agent GAE reuses the (verified) scalar GAE independently.
"""

import types

import torch


# ---- action-vector layout used by FalconDecoupledManager (name-based split) ----
# In MuJoCo dof_names order: legs 0-11, waist 12-14, arms 15-28. The env builds
# lower_action_indices = [legs..., waist...] and upper_action_indices = [arms...].
_LOWER_IDX = list(range(0, 15))   # 12 leg + 3 waist
_UPPER_IDX = list(range(15, 29))  # 14 arms


def _bare_algo(lower_idx=_LOWER_IDX, upper_idx=_UPPER_IDX, device="cpu"):
    """A PPODecoupled instance with just enough state for the permutation + wrapper
    logic, built without running __init__ (which needs a real env)."""
    from wbt_training.agents.ppo_decoupled import PPODecoupled

    algo = PPODecoupled.__new__(PPODecoupled)
    algo.device = torch.device(device)
    algo.num_act = len(lower_idx) + len(upper_idx)
    algo.action_indices = {
        "lower": torch.tensor(lower_idx, dtype=torch.long),
        "upper": torch.tensor(upper_idx, dtype=torch.long),
    }
    algo.num_act_agent = {"lower": len(lower_idx), "upper": len(upper_idx)}
    # Build the inverse permutation exactly as _init_config does.
    perm = torch.cat([algo.action_indices["lower"], algo.action_indices["upper"]])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    algo.action_inv_perm = inv
    return algo


def test_assemble_action_places_slices_in_joint_order_contiguous():
    """With the contiguous [0:15]/[15:29] split, assemble is the identity concat."""
    algo = _bare_algo()
    a_lower = torch.arange(15, dtype=torch.float).view(1, 15)
    a_upper = torch.arange(100, 114, dtype=torch.float).view(1, 14)
    full = algo._assemble_action(a_lower, a_upper)
    assert full.shape == (1, 29)
    expected = torch.cat([a_lower, a_upper], dim=-1)
    assert torch.allclose(full, expected)


def test_assemble_action_scatters_correctly_for_scattered_split():
    """The real risk: a SCATTERED (non-contiguous) action layout. Each per-agent
    entry must land on its own joint index — a transpose/gather bug would cross joints.
    Here lower owns even joint indices, upper owns odd ones."""
    lower_idx = list(range(0, 29, 2))   # 15 even indices: 0,2,...,28
    upper_idx = list(range(1, 29, 2))   # 14 odd indices: 1,3,...,27
    algo = _bare_algo(lower_idx=lower_idx, upper_idx=upper_idx)
    # Tag each agent's entry with its DESTINATION joint index so we can verify placement.
    a_lower = torch.tensor([[float(j) for j in lower_idx]])
    a_upper = torch.tensor([[float(j) for j in upper_idx]])
    full = algo._assemble_action(a_lower, a_upper)
    # full[:, j] must equal j (each slot got the value tagged with its own joint idx).
    assert torch.allclose(full[0], torch.arange(29, dtype=torch.float)), full[0]


def test_inverse_permutation_round_trips():
    """cat[inv_perm] then gather back by perm returns the original cat (bijection)."""
    lower_idx = [0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14]  # arbitrary scramble
    upper_idx = [i for i in range(29) if i not in lower_idx]
    algo = _bare_algo(lower_idx=lower_idx, upper_idx=upper_idx)
    cat = torch.randn(4, 29)
    full = cat[:, algo.action_inv_perm]  # cat(slice order) -> joint order
    perm = torch.cat([algo.action_indices["lower"], algo.action_indices["upper"]])
    back = full[:, perm]  # joint order -> slice order
    assert torch.allclose(back, cat, atol=1e-6)


def test_onnx_wrapper_emits_29d_in_joint_order():
    """The fused dual-head wrapper must run both actors on the shared obs and emit a
    29-D action in env joint order (deploy contract). Uses stub actors that return
    constant per-slice tags so placement is checkable."""
    algo = _bare_algo()

    class _StubActor:
        def __init__(self, tag_base, n):
            self._out = torch.arange(tag_base, tag_base + n, dtype=torch.float)

        def act_inference(self, state):
            b = state["actor_obs"].shape[0]
            return self._out.unsqueeze(0).expand(b, -1)

    algo.actors = {"lower": _StubActor(0, 15), "upper": _StubActor(100, 14)}
    algo.actor_obs_normalizer = torch.nn.Identity()
    algo.empirical_normalization = False
    wrapper = algo.actor_onnx_wrapper
    obs = torch.zeros(2, 8)
    out = wrapper(obs)
    assert out.shape == (2, 29)
    # Contiguous split → [lower 0..14, upper 100..113] in joint order.
    expected = torch.cat([torch.arange(0, 15.0), torch.arange(100, 114.0)]).unsqueeze(0).expand(2, -1)
    assert torch.allclose(out, expected), out[0]


def test_onnx_wrapper_scattered_split_places_by_joint():
    """Fused wrapper with a scattered split: verify each actor output lands on its
    own joint index (the deploy-side analogue of the training scatter test)."""
    lower_idx = list(range(0, 29, 2))
    upper_idx = list(range(1, 29, 2))
    algo = _bare_algo(lower_idx=lower_idx, upper_idx=upper_idx)

    class _TagActor:
        def __init__(self, dest_indices):
            self._out = torch.tensor([float(j) for j in dest_indices])

        def act_inference(self, state):
            b = state["actor_obs"].shape[0]
            return self._out.unsqueeze(0).expand(b, -1)

    algo.actors = {"lower": _TagActor(lower_idx), "upper": _TagActor(upper_idx)}
    algo.actor_obs_normalizer = torch.nn.Identity()
    algo.empirical_normalization = False
    out = algo.actor_onnx_wrapper(torch.zeros(1, 8))
    assert torch.allclose(out[0], torch.arange(29, dtype=torch.float)), out[0]


def test_get_inference_policy_matches_wrapper():
    """get_inference_policy and the ONNX wrapper must produce the SAME 29-D action
    (train/deploy inference parity)."""
    algo = _bare_algo()

    class _StubActor:
        def __init__(self, tag_base, n):
            self._out = torch.arange(tag_base, tag_base + n, dtype=torch.float)

        def act_inference(self, state):
            b = state["actor_obs"].shape[0]
            return self._out.unsqueeze(0).expand(b, -1)

        def eval(self):
            pass

        def to(self, *_a, **_k):
            return self

    algo.actors = {"lower": _StubActor(0, 15), "upper": _StubActor(100, 14)}
    algo.actor_obs_normalizer = types.SimpleNamespace(eval=lambda: None, to=lambda *a, **k: None)
    algo.empirical_normalization = False
    algo._normalize_actor_obs = lambda obs, update=True: obs  # identity
    policy = algo.get_inference_policy()
    obs = {"actor_obs": torch.zeros(3, 8)}
    a_policy = policy(obs)
    a_wrapper = algo.actor_onnx_wrapper(obs["actor_obs"])
    assert torch.allclose(a_policy, a_wrapper), "inference policy must match ONNX wrapper"


def test_per_agent_gae_shapes_and_independence():
    """Per-agent GAE = the (verified) scalar GAE called per agent. Confirm shapes and
    that different reward streams give different advantages."""
    from wbt_training.agents.ppo_decoupled import PPODecoupled

    algo = PPODecoupled.__new__(PPODecoupled)
    algo.device = torch.device("cpu")
    algo.is_multi_gpu = False
    algo.config = types.SimpleNamespace(gamma=0.99, lam=0.95)
    T, N = 5, 4
    last_v = torch.zeros(N, 1)
    values = torch.zeros(T, N, 1)
    dones = torch.zeros(T, N, 1, dtype=torch.bool)
    rew_lower = torch.ones(T, N, 1)
    rew_upper = torch.full((T, N, 1), 2.0)
    ret_l, adv_l = algo._compute_returns_and_advantages(last_v, values, dones, rew_lower)
    ret_u, adv_u = algo._compute_returns_and_advantages(last_v, values, dones, rew_upper)
    assert ret_l.shape == (T, N, 1) and adv_l.shape == (T, N, 1)
    # Different rewards → different (unnormalized) returns.
    assert not torch.allclose(ret_l, ret_u)
    # Advantages are normalized (≈ zero mean, unit std) per agent.
    assert abs(adv_l.mean().item()) < 1e-5

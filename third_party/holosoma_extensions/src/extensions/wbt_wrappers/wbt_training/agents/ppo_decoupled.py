"""FALCON-style dual-agent PPO (`PPODecoupled`) — additive, extension-local.

Two independent actors + two independent critics over ONE shared observation,
emitting disjoint action slices that are scattered into the single 29-D env action.
Per-agent GAE / advantage-normalization / KL-adaptive-LR / optimizers. This mirrors
FALCON's `PPOMultiActorCritic` but on the holosoma RSL-RL-style stack, WITHOUT
touching the monolithic `holosoma.agents.ppo.ppo.PPO` (we subclass it to reuse its
export/metadata/eval scaffolding + `EmpiricalNormalization`, and override only the
multi-actor pieces).

Agent split (from the env's name-based maps, `FalconDecoupledManager`):
  * lower — 15 DOF (12 leg + 3 waist), velocity-command locomotion reward group
  * upper — 14 DOF (arms), 2pt + F^cmd + TCP manipulation reward group
Per-agent reward comes from `env.reward_manager.group_reward_buf["lower"/"upper"]`
(the `DualGroupRewardManager`); the returned scalar env reward is unused for GAE.

Actions are DIRECT joint targets (no residual-on-ref), so the fused ONNX stays
transparent at deploy: one `actor_obs` in → both actors run → slices scattered into
one 29-D `actions` out, consumed unchanged by the standard PD law
`q_target = actions·scale + default_dof_pos`. No runtime motion input at the action
layer.

Config: `PPODecoupledConfig` (config_types/ppo_decoupled.py) — per-agent
`DecoupledAgentConfig` (network + LR + KL + noise) + shared PPO hyperparameters.
"""

from __future__ import annotations

import torch
from loguru import logger
from torch import nn

from holosoma.agents.modules.module_utils import setup_ppo_actor_module, setup_ppo_critic_module
from holosoma.agents.ppo.ppo import PPO, EmpiricalNormalization
from holosoma.utils.helpers import instantiate

_AGENTS = ("lower", "upper")


class PPODecoupled(PPO):
    """Two-actor / two-critic PPO with disjoint action slices over a shared obs."""

    # ------------------------------------------------------------------ config
    def _init_config(self) -> None:
        # Obs dims / history (shared by both agents) — same source as PPO.
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            "actor_obs": self.env.observation_manager.cfg.groups["actor_obs"].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups["critic_obs"].history_length,
        }

        # Total env action width (29) + per-agent disjoint slices from the env.
        self.num_act = self.env.robot_config.actions_dim
        try:
            lower_idx = self.env.lower_action_indices
            upper_idx = self.env.upper_action_indices
        except AttributeError as e:
            raise AttributeError(
                "PPODecoupled requires the env to expose `lower_action_indices` / "
                "`upper_action_indices` (use env_class FalconDecoupledManager)."
            ) from e
        self.action_indices = {
            "lower": lower_idx.to(self.device).long(),
            "upper": upper_idx.to(self.device).long(),
        }
        self.num_act_agent = {a: int(self.action_indices[a].numel()) for a in _AGENTS}
        assert self.num_act_agent["lower"] + self.num_act_agent["upper"] == self.num_act, (
            f"lower+upper action slices ({self.num_act_agent}) must sum to num_act={self.num_act}"
        )
        # Inverse permutation: cat[lower||upper] (in slice order) → env joint order.
        # out[:, j] = cat[:, inv_perm[j]]. Built once; ONNX-traceable Gather.
        perm = torch.cat([self.action_indices["lower"], self.action_indices["upper"]])  # joint idx per cat position
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=self.device, dtype=perm.dtype)
        self.action_inv_perm = inv_perm  # (num_act,)

        # Per-agent LR state (adapted independently).
        self.lr = {}
        for a in _AGENTS:
            ac = getattr(self.config, a)
            self.lr[a] = {
                "actor": ac.actor_learning_rate,
                "critic": ac.critic_learning_rate,
                "max_actor": ac.max_actor_learning_rate or max(ac.actor_learning_rate, 1e-2),
                "min_actor": ac.min_actor_learning_rate or min(ac.actor_learning_rate, 1e-5),
                "max_critic": ac.max_critic_learning_rate or max(ac.critic_learning_rate, 1e-2),
                "min_critic": ac.min_critic_learning_rate or min(ac.critic_learning_rate, 1e-5),
            }

        self.use_symmetry = False  # dual-agent recipe never uses symmetry augmentation
        self.empirical_normalization = self.config.empirical_normalization
        # Obs-group keys come from each agent's module input_dim (shared across agents,
        # same as the monolithic PPO's _init_obs_keys reading module_dict.actor/critic).
        self.actor_obs_keys = self.config.lower.actor.input_dim
        self.critic_obs_keys = self.config.lower.critic.input_dim

    # -------------------------------------------------------------- build nets
    def _setup_models_and_optimizer(self):
        self.actors: dict[str, nn.Module] = {}
        self.critics: dict[str, nn.Module] = {}
        self.actor_optimizers: dict[str, torch.optim.Optimizer] = {}
        self.critic_optimizers: dict[str, torch.optim.Optimizer] = {}

        for a in _AGENTS:
            ac = getattr(self.config, a)
            self.actors[a] = setup_ppo_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=ac.actor,
                num_actions=self.num_act_agent[a],  # THIS agent's slice width (15 / 14)
                init_noise_std=ac.init_noise_std,
                device=self.device,
                history_length=self.algo_history_length_dict,
            )
            self.critics[a] = setup_ppo_critic_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=ac.critic,
                device=self.device,
                history_length=self.algo_history_length_dict,
            )

        # ONE shared obs normalizer per group (obs is shared across agents; required
        # for the single-obs fused ONNX contract).
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        if self.is_multi_gpu:
            self._synchronize_model_weights()

        for a in _AGENTS:
            ac = getattr(self.config, a)
            self.actor_optimizers[a] = instantiate(
                ac.actor_optimizer, params=self.actors[a].parameters(), lr=self.lr[a]["actor"]
            )
            self.critic_optimizers[a] = instantiate(
                ac.critic_optimizer, params=self.critics[a].parameters(), lr=self.lr[a]["critic"]
            )

        # Representative aliases so the reused PPO.export's `self.actor.training`
        # check + any incidental base references resolve to a real module. All
        # training/eval/onnx paths are overridden to drive BOTH agents.
        self.actor = self.actors["lower"]
        self.critic = self.critics["lower"]

    # ------------------------------------------------------------------ storage
    def _setup_storage(self):
        from holosoma.agents.modules.data_utils import RolloutStorage

        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        self.storage.register("actor_obs", shape=(self._get_obs_dim(self.actor_obs_keys),), dtype=torch.float)
        self.storage.register("critic_obs", shape=(self._get_obs_dim(self.critic_obs_keys),), dtype=torch.float)
        self.storage.register("dones", shape=(1,), dtype=torch.bool)
        for a in _AGENTS:
            na = self.num_act_agent[a]
            for key, shape in (
                (f"actions_{a}", (na,)),
                (f"rewards_{a}", (1,)),
                (f"values_{a}", (1,)),
                (f"returns_{a}", (1,)),
                (f"advantages_{a}", (1,)),
                (f"actions_log_prob_{a}", (1,)),
                (f"action_mean_{a}", (na,)),
                (f"action_sigma_{a}", (na,)),
            ):
                self.storage.register(key, shape=shape, dtype=torch.float)

    # ------------------------------------------------------------ train/eval mode
    def _eval_mode(self):
        for a in _AGENTS:
            self.actors[a].eval()
            self.critics[a].eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def _train_mode(self):
        for a in _AGENTS:
            self.actors[a].train()
            self.critics[a].train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    # ------------------------------------------------------------ action assembly
    def _assemble_action(self, a_lower: torch.Tensor, a_upper: torch.Tensor) -> torch.Tensor:
        """Scatter the two disjoint slices into the full 29-D env action (joint order)."""
        cat = torch.cat([a_lower, a_upper], dim=-1)  # slice order: lower||upper
        return cat[:, self.action_inv_perm]  # → env joint order

    # ------------------------------------------------------------------ rollout
    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                actor_obs = self._normalize_actor_obs(actor_obs_raw)
                critic_obs = self._normalize_critic_obs(critic_obs_raw)

                acts, logps, means, sigmas, values = {}, {}, {}, {}, {}
                for a in _AGENTS:
                    acts[a] = self.actors[a].act({"actor_obs": actor_obs})
                    logps[a] = self.actors[a].get_actions_log_prob(acts[a]).detach().unsqueeze(1)
                    means[a] = self.actors[a].action_mean.detach()
                    sigmas[a] = self.actors[a].action_std.detach()
                    values[a] = self.critics[a].evaluate({"critic_obs": critic_obs}).detach()

                full_action = self._assemble_action(acts["lower"], acts["upper"])
                obs_dict, _rewards, dones, infos = self.env.step({"actions": full_action})
                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                dones = dones.to(self.device)

                # Per-agent reward from the DualGroupRewardManager (NOT the scalar).
                group = self.env.reward_manager.group_reward_buf
                rewards = {a: group[a].detach().to(self.device) for a in _AGENTS}

                # Per-agent timeout bootstrap on the SAME final critic obs.
                final_r = {a: torch.zeros_like(rewards[a]) for a in _AGENTS}
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat(
                        [infos["final_observations"][k] for k in self.critic_obs_keys], dim=1
                    )
                    final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                    to = infos["time_outs"].to(self.device).float()
                    for a in _AGENTS:
                        fv = self.critics[a].evaluate({"critic_obs": final_critic_obs}).detach()
                        final_r[a] = final_r[a] + self.config.gamma * torch.squeeze(fv * to.unsqueeze(1), 1)

                add_kwargs = {"actor_obs": actor_obs, "critic_obs": critic_obs, "dones": dones.view(-1, 1)}
                for a in _AGENTS:
                    add_kwargs[f"actions_{a}"] = acts[a]
                    add_kwargs[f"values_{a}"] = values[a]
                    add_kwargs[f"actions_log_prob_{a}"] = logps[a]
                    add_kwargs[f"action_mean_{a}"] = means[a]
                    add_kwargs[f"action_sigma_{a}"] = sigmas[a]
                    add_kwargs[f"rewards_{a}"] = (rewards[a] + final_r[a]).view(-1, 1)
                self.storage.add(**add_kwargs)

                for a in _AGENTS:
                    self.actors[a].reset(dones)
                    self.critics[a].reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards["lower"] + rewards["upper"], dones, infos)

            # Per-agent GAE.
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
            dones_buf = self.storage["dones"].to(self.device)
            for a in _AGENTS:
                last_v = self.critics[a].evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
                returns, advantages = self._compute_returns_and_advantages(
                    last_v,
                    self.storage[f"values_{a}"].to(self.device),
                    dones_buf,
                    self.storage[f"rewards_{a}"].to(self.device),
                )
                self.storage[f"returns_{a}"] = returns
                self.storage[f"advantages_{a}"] = advantages

        return obs_dict

    # ------------------------------------------------------------------ training
    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        loss_dict = {}
        for a in _AGENTS:
            loss_dict[f"Value_{a}"] = 0.0
            loss_dict[f"Surrogate_{a}"] = 0.0
            loss_dict[f"Entropy_{a}"] = 0.0
            loss_dict[f"KL_{a}"] = 0.0
        for minibatch in generator:
            for a in _AGENTS:
                self._update_agent_step(a, minibatch, loss_dict)
        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_agent_step(self, agent: str, minibatch: dict, loss_dict: dict[str, float]):
        ppo = self._compute_ppo_loss_agent(agent, minibatch)
        self.actor_optimizers[agent].zero_grad()
        self.critic_optimizers[agent].zero_grad()
        (ppo["actor_loss"] + ppo["critic_loss"]).backward()
        if self.is_multi_gpu:
            self._reduce_parameters_agent(agent)
        nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.config.max_grad_norm)
        self.actor_optimizers[agent].step()
        self.critic_optimizers[agent].step()
        loss_dict[f"Value_{agent}"] += ppo["value_loss"].item()
        loss_dict[f"Surrogate_{agent}"] += ppo["surrogate_loss"].item()
        loss_dict[f"Entropy_{agent}"] += ppo["entropy_loss"].item()
        loss_dict[f"KL_{agent}"] += ppo["kl_mean"].item()

    def _compute_ppo_loss_agent(self, agent: str, minibatch: dict):
        actor, critic = self.actors[agent], self.critics[agent]
        ac = getattr(self.config, agent)
        actions_batch = minibatch[f"actions_{agent}"]
        target_values_batch = minibatch[f"values_{agent}"]
        advantages_batch = minibatch[f"advantages_{agent}"]
        returns_batch = minibatch[f"returns_{agent}"]
        old_log_prob = minibatch[f"actions_log_prob_{agent}"]
        old_mu = minibatch[f"action_mean_{agent}"]
        old_sigma = minibatch[f"action_sigma_{agent}"]
        actor_obs = minibatch["actor_obs"]
        critic_obs = minibatch["critic_obs"]

        actor.act({"actor_obs": actor_obs})
        value_batch = critic.evaluate({"critic_obs": critic_obs})
        log_prob_batch = actor.get_actions_log_prob(actions_batch)
        mu_batch = actor.action_mean
        sigma_batch = actor.action_std
        entropy_batch = actor.entropy

        # Per-agent adaptive-LR (independent KL target).
        if ac.desired_kl is not None and self.config.schedule == "adaptive":
            kl_mean = self._compute_kl_div(old_mu, old_sigma, mu_batch, sigma_batch)
            self._update_learning_rate_agent(agent, kl_mean, ac.desired_kl)
        else:
            kl_mean = torch.zeros((), device=self.device)

        ratio = torch.exp(log_prob_batch - torch.squeeze(old_log_prob))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - ac.entropy_coef * entropy_loss
        critic_loss = self.config.value_loss_coef * value_loss
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
        }

    def _update_learning_rate_agent(self, agent: str, kl_mean: torch.Tensor, desired_kl: float):
        st = self.lr[agent]
        if kl_mean > desired_kl * 2.0:
            st["actor"] = max(st["min_actor"], st["actor"] / 1.5)
            st["critic"] = max(st["min_critic"], st["critic"] / 1.5)
        elif kl_mean < desired_kl / 2.0 and kl_mean > 0.0:
            st["actor"] = min(st["max_actor"], st["actor"] * 1.5)
            st["critic"] = min(st["max_critic"], st["critic"] * 1.5)
        for pg in self.actor_optimizers[agent].param_groups:
            pg["lr"] = st["actor"]
        for pg in self.critic_optimizers[agent].param_groups:
            pg["lr"] = st["critic"]

    # ------------------------------------------------------------ multi-GPU
    def _reduce_parameters_agent(self, agent: str):
        grads = [
            p.grad.view(-1)
            for model in (self.actors[agent], self.critics[agent])
            for p in model.parameters()
            if p.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for model in (self.actors[agent], self.critics[agent]):
            for p in model.parameters():
                if p.grad is not None:
                    n = p.numel()
                    p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad))
                    offset += n

    def _synchronize_model_weights(self):
        for a in _AGENTS:
            for p in self.actors[a].parameters():
                torch.distributed.broadcast(p.data, src=0)
            for p in self.critics[a].parameters():
                torch.distributed.broadcast(p.data, src=0)
        logger.info(f"Synchronized dual-agent model weights across {self.gpu_world_size} GPUs")

    # ------------------------------------------------------------ save / load
    def save(self, path, infos=None):
        checkpoint_dict = {"iter": self.current_learning_iteration, "infos": infos}
        for a in _AGENTS:
            checkpoint_dict[f"actor_{a}_model_state_dict"] = self.actors[a].state_dict()
            checkpoint_dict[f"critic_{a}_model_state_dict"] = self.critics[a].state_dict()
            checkpoint_dict[f"actor_{a}_optimizer_state_dict"] = self.actor_optimizers[a].state_dict()
            checkpoint_dict[f"critic_{a}_optimizer_state_dict"] = self.critic_optimizers[a].state_dict()
        checkpoint_dict["actor_obs_normalizer_state_dict"] = (
            self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None
        )
        checkpoint_dict["critic_obs_normalizer_state_dict"] = (
            self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None
        )
        checkpoint_dict.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading dual-agent checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        for a in _AGENTS:
            self.actors[a].load_state_dict(loaded[f"actor_{a}_model_state_dict"])
            self.critics[a].load_state_dict(loaded[f"critic_{a}_model_state_dict"])
        if self.empirical_normalization and loaded.get("actor_obs_normalizer_state_dict") is not None:
            self.actor_obs_normalizer.load_state_dict(loaded["actor_obs_normalizer_state_dict"])
        if self.empirical_normalization and loaded.get("critic_obs_normalizer_state_dict") is not None:
            self.critic_obs_normalizer.load_state_dict(loaded["critic_obs_normalizer_state_dict"])
        if self.config.load_optimizer:
            for a in _AGENTS:
                self.actor_optimizers[a].load_state_dict(loaded[f"actor_{a}_optimizer_state_dict"])
                self.critic_optimizers[a].load_state_dict(loaded[f"critic_{a}_optimizer_state_dict"])
                self.lr[a]["actor"] = loaded[f"actor_{a}_optimizer_state_dict"]["param_groups"][0]["lr"]
                self.lr[a]["critic"] = loaded[f"critic_{a}_optimizer_state_dict"]["param_groups"][0]["lr"]
        self.current_learning_iteration = loaded["iter"]
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    # ------------------------------------------------------------ ONNX / inference
    @property
    def actor_onnx_wrapper(self):
        """Fused dual-head wrapper: one actor_obs in → one 29-D `actions` out.

        Runs both actors (deterministic mean action) on the shared normalized obs and
        scatters the two slices into env joint order via the constant inverse
        permutation. Deploy sees a standard single-obs → 29-D-action ONNX; the
        dual-agent-ness is fully absorbed here (no residual, no runtime motion input)."""

        class DualActorWrapper(nn.Module):
            def __init__(self, actor_lower, actor_upper, normalizer, empirical, inv_perm):
                super().__init__()
                self.actor_lower = actor_lower
                self.actor_upper = actor_upper
                self.actor_obs_normalizer = normalizer
                self.empirical_normalization = empirical
                self.register_buffer("inv_perm", inv_perm)

            def forward(self, actor_obs):
                if self.empirical_normalization:
                    actor_obs = self.actor_obs_normalizer(actor_obs, update=False)
                a_lower = self.actor_lower.act_inference({"actor_obs": actor_obs})
                a_upper = self.actor_upper.act_inference({"actor_obs": actor_obs})
                cat = torch.cat([a_lower, a_upper], dim=-1)
                return cat[:, self.inv_perm]

        return DualActorWrapper(
            self.actors["lower"],
            self.actors["upper"],
            self.actor_obs_normalizer,
            self.empirical_normalization,
            self.action_inv_perm,
        )

    def get_inference_policy(self, device=None):
        for a in _AGENTS:
            self.actors[a].eval()
        self.actor_obs_normalizer.eval()
        if device is not None:
            for a in _AGENTS:
                self.actors[a].to(device)
            self.actor_obs_normalizer.to(device)

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = self._normalize_actor_obs(obs["actor_obs"], update=False)
            a_lower = self.actors["lower"].act_inference({"actor_obs": actor_obs})
            a_upper = self.actors["upper"].act_inference({"actor_obs": actor_obs})
            return self._assemble_action(a_lower, a_upper)

        return policy_fn

    # ------------------------------------------------------------ logging
    def _post_epoch_logging(self, it, loss_dict):
        extra_log_dicts = {
            "Policy": {
                "mean_noise_std_lower": self.actors["lower"].std.mean().item(),
                "mean_noise_std_upper": self.actors["upper"].std.mean().item(),
            },
        }
        loss_dict["actor_learning_rate_lower"] = self.lr["lower"]["actor"]
        loss_dict["actor_learning_rate_upper"] = self.lr["upper"]["actor"]
        loss_dict["critic_learning_rate_lower"] = self.lr["lower"]["critic"]
        loss_dict["critic_learning_rate_upper"] = self.lr["upper"]["critic"]
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

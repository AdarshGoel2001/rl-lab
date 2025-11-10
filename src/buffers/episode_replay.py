"""
Episode-aware replay buffer for world-model training.

Design highlights:
- Episode-major storage (ring of finalized episodes; per-env active accumulators)
- Memory-stable sampling (allocates O(batch_size × sequence_length), never stacks full history)
- Episode-aware windows within episodes and across boundaries (with is_first handling)
- Recency bias (timestamp percentile) and optional masks for burn-in
- Memory efficiency: observations stored as uint8; next_observations derived when needed

Usage (simplified):
    buffer = EpisodeReplayBuffer({...})
    buffer.add(trajectory={
        'observations': np.ndarray[T, E, H, W, C],
        'actions': np.ndarray[T, E, A],
        'rewards': np.ndarray[T, E],
        'dones': np.ndarray[T, E],
    })
    if buffer.ready():
        batch = buffer.sample(batch_size=16)  # observations: [B, L, H, W, C]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import time
import warnings
import numpy as np
import torch

from .base import BaseBuffer


@dataclass
class _EpisodeMeta:
    """Lightweight metadata for an episode in storage.

    Attributes:
        length: Number of timesteps (T) in episode
        timestamp: Insertion time (float, seconds)
        env_id: Optional source environment index
    """

    length: int
    timestamp: float
    env_id: Optional[int] = None


class EpisodeReplayBuffer(BaseBuffer):
    """Episode-aware replay buffer (Phase 1 skeleton).

    This class adheres to the BaseBuffer interface so it can be swapped via Hydra
    without workflow changes. It is episode-major (stores whole episodes), and the
    sampling API will return contiguous sequences that respect episode boundaries.

    Configuration (typical keys):
        capacity: int                # total timesteps capacity across episodes
        batch_size: int              # sequences per batch
        sequence_length: int         # total steps per sample (burn_in + train)
        burn_in_length: int          # prefix steps used to warm-up recurrent state
        recent_ratio: float          # fraction of batch drawn from most recent data
        num_envs: int                # number of parallel environments
        device: str                  # torch device

    Notes:
        - Observations are stored as uint8 (0..255) and converted to float32 [0,1]
          during sampling. This is not implemented in Phase 1.
        - Only `observations` are stored; `next_observations` will be derived
          during sampling. This is not implemented in Phase 1.
    """

    # -----------------------------
    # Base construction / setup
    # -----------------------------
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Initialize buffer with provided configuration.

        Args:
            config: Optional configuration dict
            **kwargs: Additional config parameters (merged with config dict)
        """
        # Merge config dict and kwargs (Hydra compatibility)
        merged_config: Dict[str, Any] = dict(config or {})
        if kwargs:
            merged_config.update(kwargs)
        config = merged_config

        # DECISION: Defer allocation to _setup_storage via BaseBuffer.__init__
        # so that capacity, device, and batch_size are standardized first.
        self.sequence_length: int = int(config.get("sequence_length", 50))
        self.burn_in_length: int = int(config.get("burn_in_length", 0))
        self.recent_ratio: float = float(config.get("recent_ratio", 0.8))
        self.num_envs: int = int(config.get("num_envs", 1))
        self.debug_mode: bool = bool(config.get("debug_mode", False))

        # Basic validation (Phase 2 scope)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not (0 <= self.burn_in_length < self.sequence_length):
            raise ValueError("burn_in_length must be in [0, sequence_length)")
        if not (0.0 <= self.recent_ratio <= 1.0):
            raise ValueError("recent_ratio must be in [0.0, 1.0]")

        # Internal validation flags and expected shapes (set on first finalized episode)
        self._validated_shapes: bool = False
        self._validated_input_range: bool = False
        self._expected_obs_shape: Optional[Tuple[int, ...]] = None
        self._expected_action_shape: Optional[Tuple[int, ...]] = None

        # Initialize BaseBuffer storage (calls _setup_storage)
        super().__init__(config)

    # BaseBuffer hook ---------------------------------------------------------
    def _setup_storage(self) -> None:
        """Allocate/initialize underlying storage.

        TODO (Phase 2):
            - Set up containers for episode-major ring buffer
            - Compute episode capacity from step capacity
            - Reset counters/indices
        """
        # NOTE: Use step-based capacity with episode-major eviction.
        # We keep deques for episodes and their metadata, and track total steps.
        # On insertion, evict oldest episodes until total_steps <= capacity.
        self._episodes: deque[Dict[str, np.ndarray]] = deque()
        self._meta: deque[_EpisodeMeta] = deque()
        self._total_steps: int = 0
        # DECISION: Avoid precomputing episode capacity; episode lengths vary.
        # Evict by steps for predictable memory footprint tied to `capacity`.
        # Initialize per-env active accumulators with the current num_envs hint.
        self._ensure_active_slots(int(getattr(self, "num_envs", 1) or 1))

    # -----------------------------
    # Public API methods
    # -----------------------------
    def add(self, **kwargs: Any) -> None:
        """Add trajectory data to the buffer.

        Expected usage (compatibility with existing code):
            add(trajectory={
                'observations': [T, num_envs, ...],
                'actions': [T, num_envs, A],
                'rewards': [T, num_envs],
                'dones': [T, num_envs],
                ...
            })

        TODO (Phase 3):
            - Convert float32 observations to uint8 for storage
            - Split vectorized trajectory into per-env episodes using 'dones'
            - Append complete episodes to ring buffer, evicting old ones as needed
            - Buffer incomplete episodes for continuation across adds
        """
        trajectory = kwargs.get("trajectory")
        if trajectory is None or not isinstance(trajectory, dict):
            raise ValueError("EpisodeReplayBuffer.add expects trajectory=dict")

        # Extract and validate required keys
        obs = trajectory.get("observations")
        actions = trajectory.get("actions")
        rewards = trajectory.get("rewards")
        dones = trajectory.get("dones")

        if obs is None or actions is None or rewards is None or dones is None:
            raise ValueError("trajectory must contain observations, actions, rewards, dones")

        obs = np.asarray(obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        # Expected leading dims: [T, num_envs, ...]
        if obs.ndim < 2:
            raise ValueError("observations must have shape [T, num_envs, ...]")
        T = int(obs.shape[0])
        env_dim = int(obs.shape[1])

        # Sync num_envs with incoming data
        if env_dim > self.num_envs:
            # Expand active accumulators if needed
            self._ensure_active_slots(env_dim)
            self.num_envs = env_dim

        # Validation: warn if float observations outside [0,1] on first add or in debug mode
        if (obs.dtype == np.float32 or obs.dtype == np.float64) and (self.debug_mode or not self._validated_input_range):
            omin = float(np.nanmin(obs))
            omax = float(np.nanmax(obs))
            if omin < -1e-6 or omax > 1.0 + 1e-6:
                warnings.warn(f"EpisodeReplayBuffer: observations outside [0,1] range detected: [{omin:.3g}, {omax:.3g}] – clipping for uint8 storage.")
            self._validated_input_range = True

        # Convert observations to uint8 storage if float-like; clip defensively.
        if obs.dtype == np.float32 or obs.dtype == np.float64:
            obs_uint8 = np.clip(np.rint(obs * 255.0), 0, 255).astype(np.uint8)
        elif obs.dtype == np.uint8:
            obs_uint8 = obs
        else:
            # Fallback: attempt safe cast with clipping
            obs_uint8 = np.clip(obs, 0, 255).astype(np.uint8)

        # Iterate steps and envs, append to active accumulators, finalize on done
        # NOTE: We do not include `next_observations` in storage.
        for t in range(T):
            for e in range(env_dim):
                self._active_obs[e].append(obs_uint8[t, e])
                self._active_actions[e].append(actions[t, e])
                self._active_rewards[e].append(rewards[t, e])
                d = bool(dones[t, e])
                self._active_dones[e].append(d)
                # TODO: Track active steps toward capacity if enforcing strict caps
                if d:
                    # Finalize this env's episode
                    episode = self._finalize_active_episode(e)
                    if episode is not None:
                        # Validate shapes on first finalized episode or in debug mode
                        self._on_episode_finalized_validate(episode)
                        self._append_episode(episode, env_id=e)
                        # After appending, accumulators are reset for env e
        # NOTE: Incomplete episodes remain in active accumulators for future adds.
        # Soft capacity check including active steps (warn-only)
        self._soft_check_active_capacity()

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample episode-respecting contiguous sequences (L = sequence_length).

        Returns a dict of PyTorch tensors on the configured device. The result will
        eventually include:
            - observations: [L, B, H, W, C] float32 in [0,1]
            - next_observations: [L, B, H, W, C] (derived via shift)
            - actions: [L, B, A]
            - rewards: [L, B]
            - dones: [L, B] (float/bool)
            - is_first: [L, B] (bool)
            - burn_in_mask: [L, B] (bool)
            - train_mask: [L, B] (bool)

        TODO (Phase 4):
            - Enumerate valid (episode_idx, start_t) windows
            - Uniform sampling for correctness (no recency bias yet)
            - Extract only requested windows (no whole-buffer materialization)

        TODO (Phase 5):
            - Add recency bias selection
            - Add burn-in/train masks
        """
        if batch_size is None:
            batch_size = int(self.batch_size)

        # Enumerate valid candidates (finalized + active, within + cross)
        candidates = self._enumerate_candidates()
        if not candidates:
            raise ValueError("No valid sequences available to sample")

        L = self.sequence_length

        # Recency bias selection (timestamp percentile). Active considered recent.
        chosen = self._sample_with_recency_bias(candidates, batch_size)

        # Option A + Y + P: list → stack → single conversion → single transfer
        obs_windows: List[np.ndarray] = []
        act_windows: List[np.ndarray] = []
        rew_windows: List[np.ndarray] = []
        done_windows: List[np.ndarray] = []
        is_first_windows: List[np.ndarray] = []

        # Snapshot episodes into a list for O(1) indexing
        episodes = list(self._episodes)

        for c in chosen:
            kind = c["kind"]
            if kind == "within_finalized":
                ep = episodes[c["ep_idx"]]
                t0 = c["start"]
                t1 = t0 + L
                obs_windows.append(ep["obs"][t0:t1])
                act_windows.append(ep["actions"][t0:t1])
                rew_windows.append(ep["rewards"][t0:t1])
                done_windows.append(ep["dones"][t0:t1])
                is_first_windows.append(ep["is_first"][t0:t1])
            elif kind == "within_active":
                e = c["env_id"]
                t0 = c["start"]
                t1 = t0 + L
                # Build small arrays from lists for this window
                obs_slice = np.stack(self._active_obs[e][t0:t1], axis=0).astype(np.uint8)
                act_slice = np.stack(self._active_actions[e][t0:t1], axis=0).astype(np.float32, copy=False)
                rew_slice = np.stack(self._active_rewards[e][t0:t1], axis=0).astype(np.float32, copy=False)
                done_slice = np.asarray(self._active_dones[e][t0:t1], dtype=bool)
                # is_first True only if starting at 0
                is_first = np.zeros(L, dtype=bool)
                if t0 == 0:
                    is_first[0] = True
                obs_windows.append(obs_slice)
                act_windows.append(act_slice)
                rew_windows.append(rew_slice)
                done_windows.append(done_slice)
                is_first_windows.append(is_first)
            elif kind == "cross_finalized_finalized":
                i1 = c["ep_idx"]
                i2 = c["next_ep_idx"]
                t0 = c["start"]
                ep1 = episodes[i1]
                ep2 = episodes[i2]
                part1 = ep1["obs"][t0:]
                len1 = part1.shape[0]
                need2 = L - len1
                part2 = ep2["obs"][:need2]
                obs_windows.append(np.concatenate([part1, part2], axis=0))

                a1 = ep1["actions"][t0:]
                a2 = ep2["actions"][:need2]
                act_windows.append(np.concatenate([a1, a2], axis=0))

                r1 = ep1["rewards"][t0:]
                r2 = ep2["rewards"][:need2]
                rew_windows.append(np.concatenate([r1, r2], axis=0))

                d1 = ep1["dones"][t0:]
                d2 = ep2["dones"][:need2]
                done_windows.append(np.concatenate([d1, d2], axis=0))

                f1 = ep1["is_first"][t0:]
                f2 = ep2["is_first"][:need2]
                # Ensure boundary is marked as start of new episode
                if need2 > 0:
                    f2 = f2.copy()
                    f2[0] = True
                is_first_windows.append(np.concatenate([f1, f2], axis=0))
            elif kind == "cross_finalized_active":
                i1 = c["ep_idx"]
                e = c["env_id"]
                t0 = c["start"]
                ep1 = episodes[i1]

                part1 = ep1["obs"][t0:]
                len1 = part1.shape[0]
                need2 = L - len1
                part2 = np.stack(self._active_obs[e][:need2], axis=0).astype(np.uint8)
                obs_windows.append(np.concatenate([part1, part2], axis=0))

                a1 = ep1["actions"][t0:]
                a2 = np.stack(self._active_actions[e][:need2], axis=0).astype(np.float32, copy=False)
                act_windows.append(np.concatenate([a1, a2], axis=0))

                r1 = ep1["rewards"][t0:]
                r2 = np.stack(self._active_rewards[e][:need2], axis=0).astype(np.float32, copy=False)
                rew_windows.append(np.concatenate([r1, r2], axis=0))

                d1 = ep1["dones"][t0:]
                # Active segment has no terminal flag yet
                d2 = np.zeros((need2,), dtype=bool)
                done_windows.append(np.concatenate([d1, d2], axis=0))

                f1 = ep1["is_first"][t0:]
                f2 = np.zeros((need2,), dtype=bool)
                if need2 > 0:
                    f2[0] = True
                is_first_windows.append(np.concatenate([f1, f2], axis=0))
            else:
                raise RuntimeError(f"Unknown candidate kind: {kind}")

        # Stack to [B, L, ...]
        obs_uint8 = np.stack(obs_windows, axis=0)                # [B, L, H, W, C]
        actions_np = np.stack(act_windows, axis=0).astype(np.float32, copy=False)  # [B, L, A]
        rewards_np = np.stack(rew_windows, axis=0).astype(np.float32, copy=False)  # [B, L]
        dones_np = np.stack(done_windows, axis=0).astype(np.bool_, copy=False)     # [B, L]
        is_first_np = np.stack(is_first_windows, axis=0).astype(np.bool_, copy=False)  # [B, L]

        # Convert observations once and transfer to torch
        obs_float = (obs_uint8.astype(np.float32) / 255.0)

        # Build CPU tensors that own their storage (no numpy sharing, no device transfer here)
        batch: Dict[str, torch.Tensor] = {
            "observations": torch.from_numpy(obs_float.copy()),
            "actions": torch.from_numpy(actions_np.copy()),
            "rewards": torch.from_numpy(rewards_np.copy()),
            "dones": torch.from_numpy(dones_np.copy()),
            "is_first": torch.from_numpy(is_first_np.copy()),
        }

        # For compatibility with existing workflows (keep on CPU)
        batch["sequence_length"] = torch.tensor(L)
        # Stride is conceptually 1 for enumerated contiguous windows in this implementation
        batch["sequence_stride"] = torch.tensor(1)

        # Masks if configured (CPU tensors)
        if self.burn_in_length > 0:
            B = actions_np.shape[0]
            burn_in_mask, train_mask = self._make_masks(L, B)
            batch["burn_in_mask"] = burn_in_mask
            batch["train_mask"] = train_mask
        return batch

    def ready(self) -> bool:
        """Return True if there is enough data to sample a batch.

        Hybrid policy: quick finalized within-episode count, then full candidate enumeration.
        """
        try:
            L = self.sequence_length
            need = int(self.batch_size)
            # Quick within-finalized count
            quick = 0
            for ep in self._episodes:
                obs = ep.get("obs")
                if isinstance(obs, np.ndarray) and obs.shape[0] >= L:
                    quick += (obs.shape[0] - L + 1)
                    if quick >= need:
                        return True
            # Fallback to full candidates (includes active + cross)
            return len(self._enumerate_candidates()) >= need
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all stored data and reset indices.

        TODO (Phase 2/6): Reset ring buffer state and counters.
        """
        self._episodes.clear()
        self._meta.clear()
        self._total_steps = 0

    def __len__(self) -> int:
        """Total timesteps currently stored across all episodes.

        TODO (Phase 2): Track and return accurate count.
        """
        return int(self._total_steps)

    # -----------------------------
    # Introspection / debug helpers
    # -----------------------------
    def num_episodes(self) -> int:
        """Number of complete episodes stored.

        TODO (Phase 2): Return actual episode count.
        """
        return int(len(self._episodes))

    def stats(self) -> Dict[str, Any]:
        """Return buffer statistics helpful for debugging and logging.

        Keys (eventually):
            total_steps, num_episodes, memory_gb, oldest_timestamp, newest_timestamp,
            mean_episode_length

        TODO (Phase 6): Compute real stats.
        """
        # NOTE: Rough memory estimate based on stored observation arrays only.
        # Actions/rewards/dones are comparatively tiny for Atari-scale frames.
        obs_bytes = 0
        newest = 0.0
        oldest = 0.0
        mean_len = 0.0
        if self._episodes:
            lengths = [m.length for m in self._meta]
            mean_len = float(np.mean(lengths)) if lengths else 0.0
            newest = max((m.timestamp for m in self._meta), default=0.0)
            oldest = min((m.timestamp for m in self._meta), default=0.0)
            for ep in self._episodes:
                obs = ep.get("obs")
                if isinstance(obs, np.ndarray):
                    obs_bytes += obs.nbytes
        # Include active obs rough size
        if hasattr(self, "_active_obs"):
            for env_list in self._active_obs:
                for frame in env_list:
                    if isinstance(frame, np.ndarray):
                        obs_bytes += frame.nbytes
        mem_gb = obs_bytes / (1024 ** 3)
        return {
            "total_steps": int(self.__len__()),
            "num_episodes": int(self.num_episodes()),
            "memory_gb": float(mem_gb),
            "oldest_timestamp": float(oldest),
            "newest_timestamp": float(newest),
            "mean_episode_length": float(mean_len),
        }

    # -----------------------------
    # Checkpointing hooks (BaseBuffer)
    # -----------------------------
    def _save_buffer_state(self) -> Dict[str, Any]:
        """Save buffer-specific state for checkpointing.

        TODO (Phase 6): Serialize episodes and metadata efficiently.
        """
        # NOTE: Phase 2 – do not serialize frames to keep patch minimal.
        # Provide lightweight metadata so a caller knows buffer was present.
        return {
            "total_steps": self._total_steps,
            "num_episodes": len(self._episodes),
            "sequence_length": self.sequence_length,
            "burn_in_length": self.burn_in_length,
            "recent_ratio": self.recent_ratio,
        }

    def _load_buffer_state(self, state: Dict[str, Any]) -> None:
        """Load buffer-specific state from a checkpoint payload.

        TODO (Phase 6): Reconstruct episodes and metadata.
        """
        # NOTE: Phase 2 – ignore serialized content; start clean.
        _ = state
        self.clear()

    # -----------------------------
    # Internal helpers (stubs for future phases)
    # -----------------------------
    def _compute_episode_capacity(self, capacity_steps: int) -> int:
        """Compute max number of episodes we can store given step capacity.

        TODO (Phase 2): Decide a policy (e.g., upper bound by min episode length).
        """
        # DECISION: Not used – we evict by total steps, not episode count.
        # Provide a conservative upper bound to assist sizing if needed.
        if capacity_steps <= 0:
            return 0
        return max(1, capacity_steps // max(1, self.sequence_length))

    def _split_trajectory_into_episodes(self, trajectory: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Split a vectorized trajectory into per-env episodes using `dones`.

        TODO (Phase 3): Implement proper splitting and handling of incomplete episodes.
        """
        # TODO: Phase 3 optional helper – not used in accumulator-based path.
        # A possible implementation would:
        #  - For each env, find indices where dones are True
        #  - Slice [start:done_idx+1] segments and create episode dicts
        #  - Return the list of episodes
        raise NotImplementedError("Use active-accumulator path; this helper is optional")

    def _enumerate_candidates(self) -> List[Dict[str, Any]]:
        """Enumerate candidate windows:
        - within_finalized: single-episode windows from finalized episodes
        - within_active: single-episode windows from active accumulators
        - cross_finalized_finalized: tail of ep_i + head of next finalized episode for same env
        - cross_finalized_active: tail of ep_i + head of active episode for same env
        """
        L = self.sequence_length
        candidates: List[Dict[str, Any]] = []

        episodes = list(self._episodes)
        meta = list(self._meta)

        # Within-finalized
        for i, ep in enumerate(episodes):
            obs = ep.get("obs")
            if not isinstance(obs, np.ndarray) or obs.ndim < 1:
                continue
            T = int(obs.shape[0])
            if T >= L:
                for t0 in range(0, T - L + 1):
                    candidates.append({
                        "kind": "within_finalized",
                        "ep_idx": i,
                        "env_id": meta[i].env_id,
                        "start": t0,
                        "timestamp": meta[i].timestamp,
                    })

        # Map env -> ordered episode indices
        env_to_indices: Dict[int, List[int]] = {}
        for i, m in enumerate(meta):
            env_id = int(m.env_id) if m.env_id is not None else -1
            env_to_indices.setdefault(env_id, []).append(i)

        # Cross finalized->finalized
        for env_id, idxs in env_to_indices.items():
            for p in range(len(idxs) - 1):
                i1 = idxs[p]
                i2 = idxs[p + 1]
                ep1 = episodes[i1]
                ep2 = episodes[i2]
                T1 = int(ep1["obs"].shape[0])
                T2 = int(ep2["obs"].shape[0])
                if T1 <= 0 or T2 <= 0:
                    continue
                # require at least 1 step from ep2
                max_from_ep1 = L - 1
                start_min = max(0, T1 - max_from_ep1)
                for t0 in range(start_min, T1):
                    len1 = T1 - t0
                    need2 = L - len1
                    if need2 <= T2:
                        candidates.append({
                            "kind": "cross_finalized_finalized",
                            "ep_idx": i1,
                            "next_ep_idx": i2,
                            "env_id": env_id,
                            "start": t0,
                            "timestamp": max(meta[i1].timestamp, meta[i2].timestamp),
                        })

        # Within-active and Cross finalized->active
        for env_id in range(max(self.num_envs, 1)):
            active_len = len(self._active_obs[env_id]) if env_id < len(self._active_obs) else 0
            if active_len >= L:
                for t0 in range(0, active_len - L + 1):
                    candidates.append({
                        "kind": "within_active",
                        "env_id": env_id,
                        "start": t0,
                        "timestamp": time.time(),  # treat as recent
                    })
            # Cross with finalized tail if possible
            # Find last finalized episode for this env
            idxs = env_to_indices.get(env_id, [])
            if active_len > 0 and idxs:
                i1 = idxs[-1]
                T1 = int(episodes[i1]["obs"].shape[0])
                max_from_ep1 = L - 1
                start_min = max(0, T1 - max_from_ep1)
                for t0 in range(start_min, T1):
                    len1 = T1 - t0
                    need2 = L - len1
                    if need2 <= active_len:
                        candidates.append({
                            "kind": "cross_finalized_active",
                            "ep_idx": i1,
                            "env_id": env_id,
                            "start": t0,
                            "timestamp": time.time(),  # recent
                        })

        return candidates

    def _sample_with_recency_bias(self, candidates: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """Sample positions with recency bias using timestamp percentile.

        Active candidates are always considered recent.
        """
        if not candidates:
            return []

        # Collect timestamps for finalized episodes to compute threshold
        times = [c["timestamp"] for c in candidates if c["timestamp"] is not None]
        if len(times) >= 5 and 0.0 <= self.recent_ratio <= 1.0:
            threshold = float(np.percentile(times, 100.0 * (1.0 - self.recent_ratio)))
        else:
            threshold = float("-inf")  # treat everything as recent when little data

        recent_pool = [c for c in candidates if c["timestamp"] >= threshold]
        all_pool = candidates

        n_recent = int(np.ceil(batch_size * float(self.recent_ratio)))
        n_recent = min(n_recent, batch_size)
        n_other = batch_size - n_recent

        chosen: List[Dict[str, Any]] = []
        if recent_pool:
            idx = np.random.choice(len(recent_pool), size=n_recent, replace=len(recent_pool) < n_recent)
            chosen.extend(recent_pool[i] for i in idx)
        else:
            n_other = batch_size  # fallback to all

        # Fill the rest from all_pool
        if n_other > 0:
            idx = np.random.choice(len(all_pool), size=n_other, replace=len(all_pool) < n_other)
            chosen.extend(all_pool[i] for i in idx)

        return chosen

    @staticmethod
    def _make_is_first(dones: np.ndarray) -> np.ndarray:
        """Return is_first flags given per-step episode termination flags.

        TODO (Phase 3/6): Implement robustly (handle empty/edge cases).
        """
        if dones.size == 0:
            return np.zeros((0,), dtype=bool)
        flags = np.zeros_like(dones, dtype=bool)
        flags[0] = True
        return flags

    @staticmethod
    def _derive_next_obs(obs_float: torch.Tensor) -> torch.Tensor:
        """Derive next_observations by 1-step shift with last-step padding.

        TODO (Phase 4/6): Implement in torch without extra large allocations.
        """
        if obs_float.dim() < 2:
            raise ValueError("observations tensor must be at least 2D [L, ...]")
        # Shift by 1 along time L with last frame padding
        last = obs_float[-1:].clone()
        return torch.cat([obs_float[1:], last], dim=0)

    def _make_masks(self, L: int, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct burn-in and training masks of shape [L, B].

        TODO (Phase 5): Implement using `burn_in_length` and device.
        """
        burn_in = torch.zeros((L, B), dtype=torch.bool)
        train = torch.ones((L, B), dtype=torch.bool)
        b = max(0, min(self.burn_in_length, L))
        if b > 0:
            burn_in[:b] = True
            train[:b] = False
        return burn_in, train

    # -----------------------------
    # Internal storage helpers (Phase 2)
    # -----------------------------
    def _append_episode(self, episode: Dict[str, np.ndarray], *, env_id: Optional[int] = None) -> None:
        """Append an episode and evict old episodes to satisfy step capacity.

        NOTE: Assumes `episode['obs']` exists and has shape [T, ...].
        """
        obs = episode.get("obs")
        if not isinstance(obs, np.ndarray) or obs.ndim < 1:
            raise ValueError("Episode must contain 'obs' ndarray with time dimension")
        T = int(obs.shape[0])
        ts = time.time()
        self._episodes.append(episode)
        self._meta.append(_EpisodeMeta(length=T, timestamp=ts, env_id=env_id))
        self._total_steps += T
        # Evict by step capacity
        while self._total_steps > self.capacity and self._episodes:
            old = self._episodes.popleft()
            old_meta = self._meta.popleft()
            self._total_steps -= int(old_meta.length)
            # NOTE: Keep at least one episode if capacity is extremely small
            if self._total_steps <= 0 and len(self._episodes) == 0:
                self._total_steps = 0
                break

    # -----------------------------
    # Active accumulators (Phase 3)
    # -----------------------------
    def _ensure_active_slots(self, required_envs: int) -> None:
        """Ensure we have per-env accumulators up to `required_envs`.

        NOTE: Called on first add() or when num_envs grows.
        """
        if not hasattr(self, "_active_obs"):
            self._active_obs: List[List[np.ndarray]] = []
            self._active_actions: List[List[np.ndarray]] = []
            self._active_rewards: List[List[np.ndarray]] = []
            self._active_dones: List[List[bool]] = []
        current = len(self._active_obs)
        for _ in range(current, required_envs):
            self._active_obs.append([])
            self._active_actions.append([])
            self._active_rewards.append([])
            self._active_dones.append([])

    def _finalize_active_episode(self, env_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Finalize the active episode for env_id if it has any steps.

        Returns an episode dict with arrays and resets the accumulators for env.
        """
        obs_list = self._active_obs[env_id]
        if not obs_list:
            return None
        actions_list = self._active_actions[env_id]
        rewards_list = self._active_rewards[env_id]
        dones_list = self._active_dones[env_id]

        # Stack lists into arrays
        obs_arr = np.stack(obs_list, axis=0).astype(np.uint8)
        actions_arr = np.stack(actions_list, axis=0)
        rewards_arr = np.stack(rewards_list, axis=0)
        dones_arr = np.asarray(dones_list, dtype=bool)
        is_first = self._make_is_first(dones_arr)

        episode = {
            "obs": obs_arr,
            "actions": actions_arr.astype(np.float32, copy=False),
            "rewards": rewards_arr.astype(np.float32, copy=False),
            "dones": dones_arr,
            "is_first": is_first,
        }

        # Reset accumulators
        self._active_obs[env_id] = []
        self._active_actions[env_id] = []
        self._active_rewards[env_id] = []
        self._active_dones[env_id] = []

        return episode

    # -----------------------------
    # Validation and capacity guards (Phase 6)
    # -----------------------------
    def _on_episode_finalized_validate(self, episode: Dict[str, np.ndarray]) -> None:
        """Validate shapes/dtypes on first finalized episode, and in debug mode thereafter.

        Raises ValueError on shape mismatches; warns on float range issues handled earlier.
        """
        obs = episode.get("obs")
        actions = episode.get("actions")
        rewards = episode.get("rewards")
        dones = episode.get("dones")
        if not (isinstance(obs, np.ndarray) and isinstance(actions, np.ndarray) and isinstance(rewards, np.ndarray) and isinstance(dones, np.ndarray)):
            raise ValueError("EpisodeReplayBuffer: episode arrays must be numpy ndarrays")

        if not self._validated_shapes:
            # Expectation from first finalized episode
            self._expected_obs_shape = tuple(obs.shape[1:])  # [H, W, C] or feature dims
            self._expected_action_shape = tuple(actions.shape[1:])
            self._validated_shapes = True
            return

        if self.debug_mode:
            if tuple(obs.shape[1:]) != self._expected_obs_shape:
                raise ValueError(f"Observation shape mismatch: got {tuple(obs.shape[1:])}, expected {self._expected_obs_shape}")
            if tuple(actions.shape[1:]) != self._expected_action_shape:
                raise ValueError(f"Action shape mismatch: got {tuple(actions.shape[1:])}, expected {self._expected_action_shape}")
            # NaN/Inf checks on numeric arrays
            for name, arr in ("actions", actions), ("rewards", rewards):
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"EpisodeReplayBuffer: {name} contains NaN/Inf")

    def _soft_check_active_capacity(self) -> None:
        """Warn if finalized+active steps exceed capacity by a small budget.

        Budget: num_envs * sequence_length (enough to finish collecting windows safely).
        """
        if not hasattr(self, "_active_obs"):
            return
        active_steps = sum(len(lst) for lst in self._active_obs)
        total_with_active = int(self._total_steps) + int(active_steps)
        budget = int(self.capacity) + int(self.num_envs) * int(self.sequence_length)
        if total_with_active > budget:
            warnings.warn(
                f"EpisodeReplayBuffer: soft capacity exceeded (with active). total={total_with_active} > budget={budget}. "
                f"Consider reducing capacity or sequence_length if memory is constrained.")

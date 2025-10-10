"""Training loop for modular world-model paradigms."""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ...utils.checkpoint import CheckpointManager
from ...utils.config import Config, load_config, resolve_device, save_config
from ...utils.logger import create_logger
from ...utils.registry import (
    auto_import_modules,
    get_buffer,
    get_environment,
)
from ..factory import ComponentFactory


logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """Minimal-yet-extensible trainer orchestrating world-model agents."""

    def __init__(self, config: Config, experiment_dir: Optional[Path] = None) -> None:
        self.config = config
        self.device = resolve_device(config.experiment.device)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = Path("experiments") / f"{config.experiment.name}_{timestamp}"
        self.experiment_dir = Path(experiment_dir) if experiment_dir else default_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        auto_import_modules()

        self.step = 0
        self.total_episodes = 0
        self.episode_returns: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        self._last_log_step = 0
        self._last_eval_step = 0
        self._last_checkpoint_step = 0
        self.start_time: Optional[float] = None

        self._initialize_components()

        self.checkpoint_manager = CheckpointManager(
            self.experiment_dir,
            auto_save_frequency=self.config.training.checkpoint_frequency,
            max_checkpoints=5,
        )

        self.experiment_logger = create_logger(
            self.experiment_dir,
            self.config.logging.__dict__,
            self.config.to_dict(),
        )

        self._attempt_resume()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / "training.log")
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(fmt)
        logging.getLogger().addHandler(handler)

    def _initialize_components(self) -> None:
        logger.info("Initializing environments and world-model paradigm")

        env_class = get_environment(self.config.environment.wrapper)
        self.environment = env_class(self.config.environment.__dict__)

        if getattr(self.config, 'evaluation', None):
            eval_class = get_environment(self.config.evaluation.wrapper)
            self.eval_environment = eval_class(self.config.evaluation.__dict__)
        else:
            eval_cfg = self.config.environment.__dict__.copy()
            if eval_cfg.get('wrapper') == 'vectorized_gym':
                eval_cfg['wrapper'] = 'gym'
                eval_cfg.pop('num_envs', None)
                eval_cfg.pop('vectorization', None)
            eval_class = get_environment(eval_cfg['wrapper'])
            self.eval_environment = eval_class(eval_cfg)

        obs_space = self.environment.observation_space
        action_space = self.environment.action_space
        self.is_vectorized = getattr(self.environment, 'is_vectorized', False)
        self.num_envs = getattr(self.environment, 'num_envs', 1)

        component_config = copy.deepcopy(self.config.components)
        paradigm_config: Dict[str, Any] = dict(component_config)
        paradigm_config['paradigm'] = 'world_model'
        paradigm_config.setdefault('paradigm_config', {})
        paradigm_config['paradigm_config'] = dict(paradigm_config['paradigm_config'])

        paradigm_overrides = getattr(self.config, 'paradigm_config', {}) or {}
        if paradigm_overrides:
            paradigm_config['paradigm_config'].update(copy.deepcopy(paradigm_overrides))

        paradigm_config['paradigm_config'].setdefault('device', self.device)

        # Ensure nested configs exist and inject device hints
        def _config_for(key: str) -> Optional[Dict[str, Any]]:
            entry = paradigm_config.get(key)
            if entry is None:
                return None
            entry = dict(entry)
            entry.setdefault('config', {})
            entry['config'] = dict(entry['config'])
            entry['config'].setdefault('device', self.device)
            paradigm_config[key] = entry
            return entry['config']

        encoder_cfg = _config_for('encoder') or {}
        encoder_cfg.setdefault('input_dim', obs_space.shape)

        policy_cfg = _config_for('policy_head') or {}
        if action_space.discrete:
            policy_cfg.setdefault('action_dim', action_space.n)
            policy_cfg.setdefault('discrete_actions', True)
        else:
            policy_cfg.setdefault('action_dim', int(np.prod(action_space.shape)))
            policy_cfg.setdefault('discrete_actions', False)

        _config_for('representation_learner')
        _config_for('dynamics_model')
        _config_for('value_function')
        _config_for('reward_predictor')
        _config_for('observation_decoder')
        _config_for('planner')

        paradigm = ComponentFactory.create_paradigm(paradigm_config)
        paradigm.to(self.device)
        self.paradigm = paradigm

        buffer_cls = get_buffer(self.config.buffer.type)
        buffer_cfg = dict(self.config.buffer.__dict__)
        buffer_cfg['device'] = self.device
        buffer_cfg.setdefault('num_envs', self.num_envs)
        buffer_cfg.setdefault('gamma', getattr(self.config.algorithm, 'gamma', 0.99))
        self.buffer = buffer_cls(buffer_cfg)

        self._set_seeds(self.config.experiment.seed)

        self._prepare_optimizers()

        training_cfg = self.config.training
        policy_warmup = int(getattr(training_cfg, "policy_warmup_updates", 0) or 0)
        actor_warmup = int(getattr(training_cfg, "actor_warmup_updates", 0) or 0)
        critic_warmup = int(getattr(training_cfg, "critic_warmup_updates", 0) or 0)

        self.actor_warmup_updates = max(actor_warmup, policy_warmup, 0)
        self.critic_warmup_updates = max(critic_warmup, policy_warmup, 0)

        self.world_model_updates = 0

        self.current_obs = None
        self.current_dones = None

        if self.is_vectorized:
            self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        else:
            self.current_episode_return = 0.0
            self.current_episode_length = 0

    def _prepare_optimizers(self) -> None:
        alg_cfg = self.config.algorithm
        world_params = []
        world_params.extend(self.paradigm.encoder.parameters())
        world_params.extend(self.paradigm.representation_learner.parameters())
        world_params.extend(self.paradigm.dynamics_model.parameters())
        if self.paradigm.observation_decoder is not None:
            world_params.extend(self.paradigm.observation_decoder.parameters())
        if self.paradigm.reward_predictor is not None:
            world_params.extend(self.paradigm.reward_predictor.parameters())

        self.world_model_optimizer = torch.optim.Adam(
            world_params,
            lr=alg_cfg.world_model_lr,
        ) if world_params else None

        self.actor_optimizer = torch.optim.Adam(
            self.paradigm.policy_head.parameters(),
            lr=alg_cfg.actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.paradigm.value_function.parameters(),
            lr=alg_cfg.critic_lr,
        )

        self.max_grad_norm = getattr(alg_cfg, 'max_grad_norm', None)

        optimizers = {}
        if self.world_model_optimizer is not None:
            optimizers['world_model'] = self.world_model_optimizer
        optimizers['actor'] = self.actor_optimizer
        optimizers['critic'] = self.critic_optimizer
        self.paradigm.optimizers = optimizers

    def _set_seeds(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        try:
            self.environment.seed(seed)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(f"Environment seed failed: {exc}")

    def _attempt_resume(self) -> None:
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint is None:
            logger.info("No checkpoint found, starting fresh")
            return

        state = self.checkpoint_manager.restore_training_state(
            {
                'algorithm': self.paradigm,
                'buffer': self.buffer,
                'metrics': {},
                'world_model_updates': self.world_model_updates,
            },
            checkpoint,
        )
        self.step = state.get('step', 0)
        self._last_log_step = self.step
        self._last_eval_step = self.step
        self._last_checkpoint_step = self.step
        self.world_model_updates = int(state.get('world_model_updates', self.world_model_updates))
        logger.info(f"Resumed training from step {self.step}")

    # ------------------------------------------------------------------
    # Training/evaluation loops
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, float]:
        logger.info("Starting world-model training loop")
        self.start_time = time.time()

        total_steps = self.config.training.total_timesteps
        log_freq = self.config.logging.log_frequency
        eval_freq = self.config.training.eval_frequency
        ckpt_freq = self.config.training.checkpoint_frequency

        while self.step < total_steps:
            self._collect_experience()

            if self.buffer.ready():
                metrics = self._update_world_model()
                self.buffer.clear()
                self.experiment_logger.log_metrics(metrics, self.step, prefix="train")

            if self.step - self._last_log_step >= log_freq:
                self._log_training_metrics()
                self._last_log_step = self.step

            if eval_freq > 0 and self.step - self._last_eval_step >= eval_freq:
                eval_metrics = self._run_evaluation()
                self.experiment_logger.log_metrics(eval_metrics, self.step, prefix="eval")
                self._last_eval_step = self.step

            if ckpt_freq > 0 and self.step - self._last_checkpoint_step >= ckpt_freq:
                self.checkpoint_manager.save_checkpoint(self._trainer_state(), self.step)
                self._last_checkpoint_step = self.step

        final_metrics = self._log_training_metrics(final_log=True)
        self.experiment_logger.log_metrics(final_metrics, self.step, prefix="train")
        self.experiment_logger.finish()
        self.checkpoint_manager.save_checkpoint(self._trainer_state(), self.step, name="final", is_best=True)
        return final_metrics

    def _collect_experience(self) -> None:
        rollout_steps = max(self.buffer.batch_size, getattr(self.config.buffer, 'sequence_length', 1))
        collected = 0

        if self.current_obs is None:
            self.current_obs = self.environment.reset()
            if self.is_vectorized:
                self.current_dones = np.zeros(self.num_envs, dtype=bool)

        while collected < rollout_steps and self.step < self.config.training.total_timesteps:
            if self.is_vectorized:
                batch = self._collect_vector_step()
                collected += 1
                self.step += self.num_envs
            else:
                batch = self._collect_single_step()
                collected += 1
                self.step += 1

            self.buffer.add(trajectory=batch)

    def _collect_single_step(self) -> Dict[str, np.ndarray]:
        obs = self.current_obs
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        action_dist = self.paradigm.forward(obs_tensor)
        action_tensor = action_dist.sample()
        action = action_tensor.cpu().numpy().squeeze(0)
        if action.shape == ():
            env_action = action.item()
        else:
            env_action = action

        next_obs, reward, done, info = self.environment.step(env_action)

        log_prob = action_dist.log_prob(action_tensor).detach().cpu().numpy()

        trajectory = {
            'observations': np.expand_dims(np.asarray(obs), axis=0),
            'next_observations': np.expand_dims(np.asarray(next_obs), axis=0),
            'actions': np.expand_dims(np.asarray(action), axis=0),
            'rewards': np.asarray([reward]),
            'dones': np.asarray([done], dtype=bool),
            'log_probs': np.asarray([log_prob]).reshape(1, -1),
        }

        self.current_episode_return += float(reward)
        self.current_episode_length += 1

        if done:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
            self.current_episode_return = 0.0
            self.current_episode_length = 0
            self.current_obs = self.environment.reset()
        else:
            self.current_obs = next_obs

        return trajectory

    def _collect_vector_step(self) -> Dict[str, np.ndarray]:
        obs = self.current_obs
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        action_dist = self.paradigm.forward(obs_tensor)
        action_tensor = action_dist.sample()

        actions_np = action_tensor.cpu().numpy()
        if actions_np.ndim == 1:
            env_actions = actions_np
        else:
            env_actions = np.asarray(actions_np)

        next_obs, rewards, dones, infos = self.environment.step(env_actions)

        log_prob = action_dist.log_prob(action_tensor)
        if log_prob.dim() == 2:
            log_prob = log_prob.sum(dim=-1)
        log_prob_np = log_prob.detach().cpu().numpy()

        trajectory = {
            'observations': np.expand_dims(np.asarray(obs), axis=0),
            'next_observations': np.expand_dims(np.asarray(next_obs), axis=0),
            'actions': np.expand_dims(np.asarray(actions_np), axis=0),
            'rewards': np.expand_dims(np.asarray(rewards), axis=0),
            'dones': np.expand_dims(np.asarray(dones, dtype=bool), axis=0),
            'log_probs': np.expand_dims(np.asarray(log_prob_np), axis=0),
        }

        self.current_obs = next_obs
        self.current_dones = np.asarray(dones, dtype=bool)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        self.vector_episode_returns += rewards_np
        self.vector_episode_lengths += 1

        done_indices = np.where(self.current_dones)[0]
        appended_indices: Dict[int, int] = {}
        for idx in done_indices:
            episodic_return = self.vector_episode_returns[idx]
            episodic_length = self.vector_episode_lengths[idx]
            self.episode_returns.append(float(episodic_return))
            self.episode_lengths.append(int(episodic_length))
            self.total_episodes += 1
            appended_indices[idx] = len(self.episode_returns) - 1
            self.vector_episode_returns[idx] = 0.0
            self.vector_episode_lengths[idx] = 0

        if hasattr(infos, '__iter__'):
            for env_idx, (done_flag, info) in enumerate(zip(self.current_dones, infos)):
                if done_flag and isinstance(info, dict) and env_idx in appended_indices:
                    ret = info.get('episode_return', info.get('episode', {}).get('r'))
                    length = info.get('episode_length', info.get('episode', {}).get('l'))
                    pos = appended_indices[env_idx]
                    if ret is not None:
                        self.episode_returns[pos] = float(ret)
                    if length is not None:
                        self.episode_lengths[pos] = int(length)

        return trajectory

    def _update_world_model(self) -> Dict[str, float]:
        batch = self.buffer.sample()
        losses = self.paradigm.compute_loss(batch)

        zero = torch.tensor(0.0, device=self.device)
        model_terms = []
        for key in ['reconstruction_loss', 'representation_loss', 'dynamics_loss', 'reward_loss']:
            if key in losses:
                model_terms.append(losses[key])
        model_loss = sum(model_terms, zero) if model_terms else zero

        actor_active = self.world_model_updates >= self.actor_warmup_updates
        critic_active = self.world_model_updates >= self.critic_warmup_updates
        actor_mask = 1.0 if actor_active else 0.0
        critic_mask = 1.0 if critic_active else 0.0

        policy_loss_raw = losses.get('policy_loss', zero)
        entropy_key_present = 'entropy_loss' in losses
        entropy_loss_raw = losses.get('entropy_loss', zero)
        value_loss_raw = losses.get('value_loss', zero)

        applied_policy_loss = policy_loss_raw * actor_mask
        applied_entropy_loss = entropy_loss_raw * actor_mask
        applied_value_loss = value_loss_raw * critic_mask

        losses['policy_objective_raw'] = policy_loss_raw.detach()
        losses['value_objective_raw'] = value_loss_raw.detach()
        if entropy_key_present:
            losses['entropy_objective_raw'] = entropy_loss_raw.detach()

        losses['policy_loss'] = applied_policy_loss
        losses['entropy_loss'] = applied_entropy_loss
        losses['value_loss'] = applied_value_loss

        total_loss = model_loss + applied_policy_loss + applied_entropy_loss + applied_value_loss
        losses['total_loss'] = total_loss

        if self.world_model_optimizer:
            self.world_model_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        if self.max_grad_norm:
            def _clip(optimizer: Optional[torch.optim.Optimizer]) -> None:
                if optimizer is None:
                    return
                params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

            if self.world_model_optimizer:
                _clip(self.world_model_optimizer)
            if actor_active:
                _clip(self.actor_optimizer)
            if critic_active:
                _clip(self.critic_optimizer)

        if self.world_model_optimizer:
            self.world_model_optimizer.step()
        if actor_active:
            self.actor_optimizer.step()
        if critic_active:
            self.critic_optimizer.step()

        losses['actor_update_mask'] = torch.tensor(actor_mask, device=self.device)
        losses['critic_update_mask'] = torch.tensor(critic_mask, device=self.device)
        self.world_model_updates += 1

        metrics = {name: value.item() if isinstance(value, torch.Tensor) else value for name, value in losses.items() if isinstance(value, torch.Tensor)}
        metrics['total_loss'] = total_loss.item()
        metrics['world_model_updates'] = float(self.world_model_updates)
        metrics['actor_active'] = float(actor_active)
        metrics['critic_active'] = float(critic_active)
        return metrics

    def _run_evaluation(self) -> Dict[str, float]:
        self.paradigm.encoder.eval()
        self.paradigm.policy_head.eval()

        returns = []
        lengths = []
        skipped_episodes = 0
        for episode in range(self.config.training.num_eval_episodes):
            obs = self.eval_environment.reset()
            done = False
            episode_return = 0.0
            steps = 0
            skip_episode = False
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.paradigm.forward(obs_tensor)
                    if hasattr(dist, 'mode'):
                        action_tensor = dist.mode
                    elif hasattr(dist, 'mean') and not torch.isnan(dist.mean).any():
                        action_tensor = dist.mean
                    elif hasattr(dist, 'probs'):
                        action_tensor = dist.probs.argmax(dim=-1)
                    else:
                        action_tensor = dist.sample()
                action_np = action_tensor.cpu().numpy().squeeze(0)
                if np.isnan(action_np).any():
                    skipped_episodes += 1
                    logger.warning(
                        "Skipping evaluation episode %s at trainer step %s due to NaN action.",
                        episode,
                        self.step,
                    )
                    skip_episode = True
                    break
                if action_np.shape == ():
                    env_action = int(action_np.item())
                else:
                    env_action = action_np
                obs, reward, done, _ = self.eval_environment.step(env_action)
                episode_return += reward
                steps += 1
            if not skip_episode:
                returns.append(episode_return)
                lengths.append(steps)

        self.paradigm.encoder.train()
        self.paradigm.policy_head.train()

        if returns:
            metrics = {
                'return_mean': float(np.mean(returns)),
                'return_std': float(np.std(returns)),
                'return_max': float(np.max(returns)),
                'return_min': float(np.min(returns)),
                'episode_length_mean': float(np.mean(lengths)) if lengths else float('nan'),
            }
        else:
            metrics = {
                'return_mean': float('nan'),
                'return_std': float('nan'),
                'return_max': float('nan'),
                'return_min': float('nan'),
                'episode_length_mean': float('nan'),
            }

        if skipped_episodes:
            metrics['eval_skipped_episodes'] = float(skipped_episodes)

        return metrics

    def _log_training_metrics(self, final_log: bool = False) -> Dict[str, float]:
        metrics = {
            'step': float(self.step),
            'episodes': float(self.total_episodes),
        }
        if self.episode_returns:
            returns_arr = np.array(self.episode_returns, dtype=np.float32)
            metrics.update({
                'return_recent_mean': float(returns_arr.mean()),
                'return_recent_std': float(returns_arr.std()),
                'return_recent_max': float(returns_arr.max()),
                'return_recent_min': float(returns_arr.min()),
            })
        if self.episode_lengths:
            lengths_arr = np.array(self.episode_lengths, dtype=np.float32)
            metrics['episode_length_recent_mean'] = float(lengths_arr.mean())

        if self.start_time:
            elapsed = time.time() - self.start_time
            metrics['time_elapsed'] = float(elapsed)
            metrics['steps_per_second'] = float(self.step / max(elapsed, 1e-8))

        if not final_log:
            return metrics

        return metrics

    def _trainer_state(self) -> Dict[str, Any]:
        return {
            'algorithm': self.paradigm,
            'buffer': self.buffer,
            'step': self.step,
            'world_model_updates': self.world_model_updates,
        }

    def save_experiment_config(self) -> None:
        config_dir = self.experiment_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        save_config(self.config, config_dir / "config.yaml")

    def cleanup(self) -> None:
        try:
            self.environment.close()
            self.eval_environment.close()
        except Exception:  # pragma: no cover - best effort
            pass


def create_trainer_from_config(
    config_path: str,
    experiment_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> WorldModelTrainer:
    config = load_config(config_path, config_overrides)
    trainer = WorldModelTrainer(config, Path(experiment_dir) if experiment_dir else None)
    trainer.save_experiment_config()
    return trainer

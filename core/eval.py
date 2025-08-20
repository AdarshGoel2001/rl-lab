"""Evaluation harness for running policy rollouts."""

from __future__ import annotations

from typing import Any, Dict

import torch


def evaluate(agent: Any, env: Any, episodes: int = 10) -> Dict[str, float]:
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs).float()
            with torch.no_grad():
                action, _, _ = agent.act(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_reward += ep_reward
    return {"episode_reward": total_reward / episodes}



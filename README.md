# Humanoid PPO
A lightweight PyBullet + PyTorch implementation of a simplified humanoid sprint environment with smart initialization and a minimalist PPO (Proximal Policy Optimization) agent.

This project is designed for rapid prototyping and educational experiments with reinforcement learning (RL) on simulated humanoid locomotion tasks.

## 🚀 Features
- Simplified Environment:

- PyBullet-based humanoid with reduced joint control complexity.

  - Smart reset with phase-dependent initial walking poses for faster convergence.

  - Reward function combines forward velocity, stability, reference pose tracking, and low energy penalties.

- Efficient PPO Implementation:

  - Lightweight Actor-Critic network with shared layers.

  - Conservative exploration (small initial std) to avoid instability.

  - Reduced buffer size and epochs for quick experiments.

- Observation Normalization:

  - Running mean and variance tracker to stabilize learning.

- Logging & Checkpoints:

  - Logs metrics to log.txt and periodically saves models.

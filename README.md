# 🏃‍♂️ HalfCheetah PPO

A modern **PyTorch + Gymnasium** implementation of **Proximal Policy Optimization (PPO)** for continuous control in the `HalfCheetah-v4` environment.

This project is built for clarity, modularity, and fast iteration, making it suitable for research, education, and benchmarking experiments.

## 🚀 Features

### ✅ Robust PPO Core
- Actor-Critic architecture with a shared MLP torso and separate heads.
- Orthogonal weight initialization and learnable log standard deviation for action noise.
- Clipped objective and KL-based early stopping for training stability.

### 📦 Smart Buffer & GAE
- Generalized Advantage Estimation (GAE) for low-bias, low-variance learning.
- Trajectory segmentation and normalized advantages for improved convergence.
- Efficient buffer reset and batched mini-updates across multiple PPO epochs.

### ⚙️ Efficient Training Pipeline
- Modular training loop with configurable `buffer_size`, `batch_size`, and `ppo_epochs`.
- Gradient clipping and entropy regularization for stability.
- On-policy updates scheduled after each full trajectory batch.

### 💾 Logging & Checkpoints
- Logs average episode reward, length, and loss metrics.
- Auto-saves model checkpoints and full config JSON.
- Optional plotting function for training reward curves.

### 🧪 Testing & Evaluation
- Deterministic action selection during evaluation.
- Command-line selectable test mode with optional rendering.
- Summary stats for episode reward and length.

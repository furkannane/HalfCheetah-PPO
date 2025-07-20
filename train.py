import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
import json
import time
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


class PPONetwork(nn.Module):
    """Modern PPO network with separate actor and critic heads."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using modern techniques."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Initialize actor output layer with smaller weights
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, obs):
        """Forward pass returning action distribution and value."""
        features = self.shared_layers(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        action_dist = Normal(action_mean, action_std)
        
        # Critic
        value = self.critic(features)
        
        return action_dist, value.squeeze(-1)


class PPOBuffer:
    """Experience replay buffer for PPO with GAE computation."""
    
    def __init__(self, obs_dim, action_dim, buffer_size, gamma=0.99, lam=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(self, obs, action, reward, value, log_prob, done):
        """Store a single step."""
        assert self.ptr < self.buffer_size
        
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """Compute GAE and returns for the last trajectory."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute returns
        self.returns[path_slice] = self._discount_cumsum(rewards, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """Get all data and reset buffer."""
        assert self.ptr == self.buffer_size
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        data = dict(
            obs=torch.FloatTensor(self.obs),
            actions=torch.FloatTensor(self.actions),
            returns=torch.FloatTensor(self.returns),
            advantages=torch.FloatTensor(self.advantages),
            log_probs=torch.FloatTensor(self.log_probs),
        )
        
        self.ptr = 0
        self.path_start_idx = 0
        
        return data
    
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum."""
        return np.array([np.sum(x[i:] * (discount ** np.arange(len(x) - i))) for i in range(len(x))])


class PPOAgent:
    """Modern PPO agent with best practices."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Create network and optimizer
        self.network = PPONetwork(self.obs_dim, self.action_dim, config['hidden_dim'])
        self.optimizer = optim.Adam(self.network.parameters(), lr=config['learning_rate'])
        
        # Create buffer
        self.buffer = PPOBuffer(
            self.obs_dim, 
            self.action_dim, 
            config['buffer_size'],
            config['gamma'],
            config['gae_lambda']
        )
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Save config
        with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def select_action(self, obs, deterministic=False):
        """Select action using current policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_dist, value = self.network(obs_tensor)
            
            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action).sum(-1)
            
            return action.cpu().numpy()[0], value.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def update(self):
        """Update the policy using PPO."""
        data = self.buffer.get()
        
        # Convert to tensors
        obs = data['obs']
        actions = data['actions']
        returns = data['returns']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        kl_divergence = 0
        
        # Multiple epochs of optimization
        for epoch in range(self.config['ppo_epochs']):
            # Mini-batch training
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.config['batch_size']):
                end = start + self.config['batch_size']
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                action_dist, values = self.network(batch_obs)
                new_log_probs = action_dist.log_prob(batch_actions).sum(-1)
                entropy = action_dist.entropy().sum(-1).mean()
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.config['value_coef'] * value_loss - self.config['entropy_coef'] * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                # Compute KL divergence for early stopping
                kl = (batch_old_log_probs - new_log_probs).mean()
                kl_divergence += kl.item()
            
            # Early stopping based on KL divergence
            if kl_divergence / (len(obs) // self.config['batch_size']) > self.config['target_kl']:
                print(f"Early stopping at epoch {epoch + 1} due to KL divergence: {kl_divergence:.4f}")
                break
        
        return {
            'policy_loss': total_policy_loss,
            'value_loss': total_value_loss,
            'entropy': total_entropy,
            'kl_divergence': kl_divergence
        }
    
    def train(self):
        """Main training loop."""
        print("Starting PPO training...")
        print(f"Environment: {self.env.spec.id}")
        print(f"Observation space: {self.env.observation_space}")
        print(f"Action space: {self.env.action_space}")
        print("-" * 50)
        
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config['total_steps']):
            # Select action
            action, value, log_prob = self.select_action(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.buffer.store(obs, action, reward, value, log_prob, done)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Finish trajectory
                self.buffer.finish_path()
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Update policy
            if (step + 1) % self.config['buffer_size'] == 0:
                if not done:
                    # If trajectory didn't end, bootstrap value
                    _, last_value, _ = self.select_action(obs)
                    self.buffer.finish_path(last_value)
                
                # Update policy
                update_info = self.update()
                
                # Logging
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    
                    print(f"Step {step + 1:7d} | "
                          f"Reward: {avg_reward:8.2f} | "
                          f"Length: {avg_length:6.1f} | "
                          f"Policy Loss: {update_info['policy_loss']:8.4f} | "
                          f"Value Loss: {update_info['value_loss']:8.4f}")
                
                # Save checkpoint
                if (step + 1) % self.config['save_freq'] == 0:
                    self.save_checkpoint(step + 1)
        
        # Final checkpoint
        self.save_checkpoint(self.config['total_steps'])
        print("Training completed!")
    
    def save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(self.config['checkpoint_dir'], 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.network.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint


def create_config():
    """Create default configuration."""
    return {
        # Environment
        'env_name': 'HalfCheetah-v4',
        
        # Network
        'hidden_dim': 256,
        
        # Training
        'total_steps': 1000000,
        'buffer_size': 4096,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        
        # PPO specific
        'ppo_epochs': 10,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'target_kl': 0.01,
        
        # Checkpoints
        'checkpoint_dir': 'checkpoints',
        'save_freq': 40960,
    }


def train():
    """Train the PPO agent."""
    config = create_config()
    
    # Create environment
    env = gym.make(config['env_name'])
    
    # Create and train agent
    agent = PPOAgent(env, config)
    agent.train()
    
    env.close()


def test(checkpoint_path, num_episodes=1, render=True):
    """Test the trained agent."""
    config = create_config()
    
    # Create environment with rendering if requested
    render_mode = "human" if render else None
    env = gym.make(config['env_name'], render_mode=render_mode)
    
    # Create agent and load checkpoint
    agent = PPOAgent(env, config)
    checkpoint = agent.load_checkpoint(checkpoint_path)
    
    print(f"Testing agent trained for {checkpoint['step']} steps")
    print(f"Running {num_episodes} episodes...")
    print("-" * 50)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action deterministically for testing
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(0.02)  # Slow down for visualization
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:8.2f}, Length = {episode_length:4d}")
    
    print("-" * 50)
    print(f"Average Reward: {np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}")
    print(f"Average Length: {np.mean(episode_lengths):8.1f} ± {np.std(episode_lengths):6.1f}")
    
    env.close()


def plot_training_progress(checkpoint_dir):
    """Plot training progress from checkpoints."""
    checkpoints = []
    rewards = []
    steps = []
    
    # Collect data from all checkpoints
    for file in sorted(os.listdir(checkpoint_dir)):
        if file.startswith('checkpoint_') and file.endswith('.pt'):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'episode_rewards' in checkpoint and checkpoint['episode_rewards']:
                checkpoints.append(checkpoint)
                rewards.append(np.mean(checkpoint['episode_rewards']))
                steps.append(checkpoint['step'])
    
    if not checkpoints:
        print("No checkpoint data found for plotting.")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Training Steps')
    plt.ylabel('Average Episode Reward')
    plt.title('PPO Training Progress on HalfCheetah-v4')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(checkpoint_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training progress plot saved: {plot_path}")


if __name__ == "__main__":

    # ['train', 'test', 'plot']
    mode = 'test'

    checkpoint = 'checkpoints/checkpoint_983040.pt'
    
    if mode == 'train':
        train()
    elif mode == 'test':
        test(checkpoint)
    elif mode == 'plot':
        plot_training_progress('checkpoints')
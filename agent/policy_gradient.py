"""Policy Gradient RL agent for quantum circuit optimization."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from collections import deque


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128, 64]):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (3 for MultiDiscrete)
            hidden_dims: List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for each action component
        self.action_type_head = nn.Linear(input_dim, 5)  # 5 action types
        self.gate_index_head = nn.Linear(input_dim, 100)  # Max 100 gates
        self.param_head = nn.Linear(input_dim, 10)  # 10 params
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of action logits for each component
        """
        # Check for NaN or inf in input
        if torch.isnan(state).any() or torch.isinf(state).any():
            # Replace NaN/inf with zeros
            state = torch.where(torch.isnan(state) | torch.isinf(state), 
                               torch.zeros_like(state), state)
        
        features = self.shared_layers(state)
        
        # Check for NaN in features
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.where(torch.isnan(features) | torch.isinf(features),
                                  torch.zeros_like(features), features)
        
        action_type_logits = self.action_type_head(features)
        gate_index_logits = self.gate_index_head(features)
        param_logits = self.param_head(features)
        
        # Clamp logits to prevent extreme values
        action_type_logits = torch.clamp(action_type_logits, min=-10, max=10)
        gate_index_logits = torch.clamp(gate_index_logits, min=-10, max=10)
        param_logits = torch.clamp(param_logits, min=-10, max=10)
        
        return action_type_logits, gate_index_logits, param_logits


class PolicyGradientAgent:
    """Policy Gradient agent using REINFORCE algorithm."""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Training buffers
        self.saved_log_probs = []
        self.rewards = []
        
        # Metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'gate_counts': [],
            'depths': [],
            'fidelities': []
        }
    
    def select_action(self, state, training=True):
        """
        Select an action using the policy network.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action tuple
        """
        # Validate and clean state
        state = np.array(state, dtype=np.float32)
        
        # Check for NaN or inf
        if np.isnan(state).any() or np.isinf(state).any():
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize state to prevent extreme values
        state = np.clip(state, -1.0, 1.0)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # CRITICAL: Don't use no_grad() when training - we need gradients for log_probs
        if training:
            self.policy_net.train()
            # Compute forward pass WITH gradients
            action_type_logits, gate_index_logits, param_logits = self.policy_net(state_tensor)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                action_type_logits, gate_index_logits, param_logits = self.policy_net(state_tensor)
        
        # Check for NaN in logits
        if torch.isnan(action_type_logits).any() or torch.isnan(gate_index_logits).any() or torch.isnan(param_logits).any():
            # If NaN detected, use random action as fallback
            print("Warning: NaN detected in logits, using random action")
            return [
                np.random.randint(0, 5),
                np.random.randint(0, 100),
                np.random.randint(0, 10)
            ]
        
        # Create categorical distributions
        action_type_dist = distributions.Categorical(logits=action_type_logits)
        gate_index_dist = distributions.Categorical(logits=gate_index_logits)
        param_dist = distributions.Categorical(logits=param_logits)
        
        if training:
            # Sample actions
            action_type = action_type_dist.sample()
            gate_index = gate_index_dist.sample()
            param = param_dist.sample()
            
            # Save log probabilities for policy gradient
            # These MUST have gradients for backward() to work
            log_prob_type = action_type_dist.log_prob(action_type)
            log_prob_idx = gate_index_dist.log_prob(gate_index)
            log_prob_param = param_dist.log_prob(param)
            
            # Check for NaN in log probs - if NaN, don't save (will skip in update)
            if torch.isnan(log_prob_type) or torch.isnan(log_prob_idx) or torch.isnan(log_prob_param):
                # Skip this action - don't save log probs
                pass
            else:
                # Verify gradients are enabled
                if not log_prob_type.requires_grad:
                    # This shouldn't happen, but if it does, we need to enable grad
                    log_prob_type = log_prob_type.detach().requires_grad_(True)
                if not log_prob_idx.requires_grad:
                    log_prob_idx = log_prob_idx.detach().requires_grad_(True)
                if not log_prob_param.requires_grad:
                    log_prob_param = log_prob_param.detach().requires_grad_(True)
                
                self.saved_log_probs.append((log_prob_type, log_prob_idx, log_prob_param))
        else:
            # Greedy action
            action_type = action_type_logits.argmax(dim=-1)
            gate_index = gate_index_logits.argmax(dim=-1)
            param = param_logits.argmax(dim=-1)
        
        return [action_type.item(), gate_index.item(), param.item()]
    
    def store_reward(self, reward):
        """Store reward for the current step."""
        self.rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns)
        
        # Check for NaN in returns
        if torch.isnan(returns).any():
            print("Warning: NaN in returns, skipping update")
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
        
        # Normalize returns
        returns_mean = returns.mean()
        returns_std = returns.std()
        if returns_std > 1e-8:
            returns = (returns - returns_mean) / returns_std
        else:
            returns = returns - returns_mean
        
        # Clip returns to prevent extreme values
        returns = torch.clamp(returns, min=-10, max=10)
        
        # Calculate policy loss
        policy_loss = []
        valid_pairs = []
        
        # Filter out invalid log probs and match with returns
        for i, (log_prob_type, log_prob_idx, log_prob_param) in enumerate(self.saved_log_probs):
            # Check for NaN in log probs
            if torch.isnan(log_prob_type) or torch.isnan(log_prob_idx) or torch.isnan(log_prob_param):
                continue
            
            # Check if log probs require grad
            if not log_prob_type.requires_grad or not log_prob_idx.requires_grad or not log_prob_param.requires_grad:
                continue
            
            if i < len(returns):
                valid_pairs.append((log_prob_type, log_prob_idx, log_prob_param, returns[i]))
        
        if len(valid_pairs) == 0:
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
        
        # Calculate loss for valid pairs
        for log_prob_type, log_prob_idx, log_prob_param, R in valid_pairs:
            loss = -(log_prob_type + log_prob_idx + log_prob_param) * R
            policy_loss.append(loss)
        
        if len(policy_loss) == 0:
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Check for NaN in loss
        if torch.isnan(policy_loss):
            print("Warning: NaN in policy loss, skipping update")
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
        
        # Update network with gradient clipping
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in self.policy_net.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            print("Warning: NaN gradients detected, skipping update")
            self.optimizer.zero_grad()
            self.saved_log_probs = []
            self.rewards = []
            return 0.0
        
        self.optimizer.step()
        
        # Clear buffers
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)

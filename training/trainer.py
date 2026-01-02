"""Training loop for the RL agent."""

import numpy as np
import os
from datetime import datetime
import json
import copy
from environment.quantum_env import QuantumCircuitEnv
from agent.policy_gradient import PolicyGradientAgent
from circuits.circuit_library import CircuitLibrary


class Trainer:
    """Manages RL training process."""
    
    def __init__(self, config=None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or self._default_config()
        self.agent = None
        self.current_circuit = None
        self.env = None
        
        # Results storage
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Best optimized circuit tracking
        self.best_circuit = None
        self.best_reward = float('-inf')
        self.best_metrics = None
        
    def _default_config(self):
        """Default training configuration."""
        return {
            'num_episodes': 1000,
            'max_steps_per_episode': 50,
            'fidelity_threshold': 0.99,
            'learning_rate': 1e-4,  # Reduced to prevent gradient explosion
            'gamma': 0.99,
            'circuit_name': 'ghz_3',
            'checkpoint_interval': 100,
            'log_interval': 10
        }
    
    def initialize_agent(self, state_dim, action_dim):
        """Initialize the RL agent."""
        self.agent = PolicyGradientAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=self.config['learning_rate'],
            gamma=self.config['gamma']
        )
    
    def load_circuit(self, circuit_name=None):
        """Load a circuit from the library or custom file."""
        # Check if custom circuit file is provided
        custom_file = self.config.get('custom_circuit_file')
        if custom_file and os.path.exists(custom_file):
            from qiskit import QuantumCircuit
            self.current_circuit = QuantumCircuit.from_qasm_file(custom_file)
            return self.current_circuit
        
        # Otherwise load from library
        library = CircuitLibrary()
        circuits = library.get_all_circuits()
        
        if circuit_name is None:
            circuit_name = self.config['circuit_name']
        
        if circuit_name not in circuits:
            raise ValueError(f"Circuit {circuit_name} not found. Available: {list(circuits.keys())}")
        
        self.current_circuit = circuits[circuit_name]
        return self.current_circuit
    
    def create_environment(self):
        """Create the training environment."""
        if self.current_circuit is None:
            self.load_circuit()
        
        self.env = QuantumCircuitEnv(
            original_circuit=self.current_circuit,
            fidelity_threshold=self.config['fidelity_threshold'],
            max_steps=self.config['max_steps_per_episode']
        )
        
        if self.agent is None:
            state_dim = self.env.observation_space.shape[0]
            action_dim = 3  # MultiDiscrete with 3 components
            self.initialize_agent(state_dim, action_dim)
        
        return self.env
    
    def train_episode(self):
        """Train for one episode."""
        if self.env is None:
            self.create_environment()
        
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        initial_gate_count = info['gate_count']
        initial_depth = info['depth']
        
        done = False
        while not done:
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            self.agent.store_reward(reward)
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            done = terminated or truncated
        
        # Update policy after episode
        policy_loss = self.agent.update_policy()
        
        # Record metrics
        final_gate_count = info['gate_count']
        final_depth = info['depth']
        final_fidelity = info['fidelity']
        
        self.agent.training_history['episode_rewards'].append(episode_reward)
        self.agent.training_history['episode_lengths'].append(episode_length)
        self.agent.training_history['gate_counts'].append(final_gate_count)
        self.agent.training_history['depths'].append(final_depth)
        self.agent.training_history['fidelities'].append(final_fidelity)
        
        # Track best circuit (highest reward with good fidelity)
        if episode_reward > self.best_reward and final_fidelity >= self.config['fidelity_threshold']:
            self.best_reward = episode_reward
            self.best_circuit = copy.deepcopy(self.env.current_circuit)
            self.best_metrics = {
                'gate_count': final_gate_count,
                'depth': final_depth,
                'fidelity': final_fidelity,
                'gate_reduction': initial_gate_count - final_gate_count,
                'depth_reduction': initial_depth - final_depth,
                'reward': episode_reward
            }
        
        return {
            'episode': self.current_episode,
            'reward': episode_reward,
            'length': episode_length,
            'initial_gate_count': initial_gate_count,
            'final_gate_count': final_gate_count,
            'initial_depth': initial_depth,
            'final_depth': final_depth,
            'fidelity': final_fidelity,
            'gate_reduction': initial_gate_count - final_gate_count,
            'depth_reduction': initial_depth - final_depth,
            'policy_loss': policy_loss
        }
    
    def train(self, num_episodes=None):
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train (overrides config)
        """
        if num_episodes is None:
            num_episodes = self.config['num_episodes']
        
        self.is_training = True
        self.create_environment()
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Circuit: {self.config['circuit_name']}")
        print(f"Initial Gate Count: {self.current_circuit.size()}")
        print(f"Initial Depth: {self.current_circuit.depth()}")
        print("-" * 50)
        
        for episode in range(num_episodes):
            self.current_episode = episode
            episode_stats = self.train_episode()
            
            if (episode + 1) % self.config['log_interval'] == 0:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Reward: {episode_stats['reward']:.2f}")
                print(f"  Gate Count: {episode_stats['final_gate_count']} "
                      f"(reduction: {episode_stats['gate_reduction']})")
                print(f"  Depth: {episode_stats['final_depth']} "
                      f"(reduction: {episode_stats['depth_reduction']})")
                print(f"  Fidelity: {episode_stats['fidelity']:.4f}")
                print("-" * 50)
            
            if (episode + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(episode + 1)
        
        self.is_training = False
        self.save_results()
        print("Training completed!")
    
    def save_checkpoint(self, episode):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_ep{episode}_{self.session_id}.pth"
        )
        self.agent.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self):
        """Save training results."""
        results_path = os.path.join(
            self.results_dir,
            f"training_results_{self.session_id}.json"
        )
        
        # Get final metrics from training history
        final_gate_count = self.agent.training_history['gate_counts'][-1] if self.agent.training_history['gate_counts'] else self.current_circuit.size()
        final_depth = self.agent.training_history['depths'][-1] if self.agent.training_history['depths'] else self.current_circuit.depth()
        final_fidelity = self.agent.training_history['fidelities'][-1] if self.agent.training_history['fidelities'] else 1.0
        
        results = {
            'session_id': self.session_id,
            'config': self.config,
            'training_history': self.agent.training_history,
            'circuit_name': self.config['circuit_name'],
            'original_circuit': {
                'gate_count': self.current_circuit.size(),
                'depth': self.current_circuit.depth(),
                'qasm': self.current_circuit.qasm()  # Save original circuit QASM
            },
            'best_metrics': self.best_metrics
        }
        
        # Determine optimized metrics (use best if available, otherwise final)
        if self.best_circuit and self.best_metrics:
            optimized_gate_count = self.best_metrics.get('gate_count', self.best_circuit.size())
            optimized_depth = self.best_metrics.get('depth', self.best_circuit.depth())
            optimized_fidelity = self.best_metrics.get('fidelity', 1.0)
            
            # Save best circuit
            best_circuit_path = os.path.join(
                self.results_dir,
                f"best_circuit_{self.session_id}.qasm"
            )
            with open(best_circuit_path, 'w') as f:
                f.write(self.best_circuit.qasm())
            results['best_circuit_file'] = f"best_circuit_{self.session_id}.qasm"
            results['optimized_circuit'] = {
                'qasm': self.best_circuit.qasm(),  # Save optimized circuit QASM in JSON
                'gate_count': optimized_gate_count,
                'depth': optimized_depth,
                'fidelity': optimized_fidelity
            }
        else:
            # Use final metrics if no best circuit
            optimized_gate_count = final_gate_count
            optimized_depth = final_depth
            optimized_fidelity = final_fidelity
        
        # ALWAYS save optimization_results (calculated from original vs optimized)
        original_gate_count = results['original_circuit']['gate_count']
        original_depth = results['original_circuit']['depth']
        
        results['optimization_results'] = {
            'gate_count_reduction': original_gate_count - optimized_gate_count,
            'depth_reduction': original_depth - optimized_depth,
            'gate_count_reduction_percent': ((original_gate_count - optimized_gate_count) / original_gate_count * 100) if original_gate_count > 0 else 0,
            'depth_reduction_percent': ((original_depth - optimized_depth) / original_depth * 100) if original_depth > 0 else 0
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved: {results_path}")
    
    def stop_training(self):
        """Stop training gracefully."""
        self.is_training = False
        if self.agent:
            self.save_results()

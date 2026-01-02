"""Custom Gymnasium environment for quantum circuit optimization."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate, CXGate, CZGate, TGate, SGate, XGate, YGate, ZGate
import copy


class QuantumCircuitEnv(gym.Env):
    """
    Custom Gymnasium environment for optimizing quantum circuits using RL.
    
    State: Encoded quantum circuit representation
    Action: Circuit transformation operations
    Reward: Based on gate count reduction, depth reduction, and fidelity preservation
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    # Action types
    ACTION_REMOVE_GATE = 0
    ACTION_MERGE_GATES = 1
    ACTION_REPLACE_GATE = 2
    ACTION_COMMUTE_GATES = 3
    ACTION_NO_OP = 4
    
    def __init__(self, original_circuit, fidelity_threshold=0.99, max_steps=50):
        """
        Initialize the environment.
        
        Args:
            original_circuit: Original QuantumCircuit to optimize
            fidelity_threshold: Minimum fidelity to maintain (default: 0.99)
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.original_circuit = original_circuit
        self.fidelity_threshold = fidelity_threshold
        self.max_steps = max_steps
        
        # Store original metrics
        self.original_statevector = Statevector.from_instruction(original_circuit)
        self.original_gate_count = original_circuit.size()
        self.original_depth = original_circuit.depth()
        
        # Current state
        self.current_circuit = None
        self.step_count = 0
        self.last_gate_count = None
        self.last_depth = None
        
        # Define action space: [action_type, gate_index, optional_param]
        # Simplified: action_type (0-4), gate_index (0-99), param (0-9)
        self.action_space = spaces.MultiDiscrete([5, 100, 10])
        
        # Define observation space: encoded circuit state
        # Features: gate count, depth, normalized gate types, connectivity
        max_gates = 100
        max_qubits = 10
        obs_dim = 2 + max_gates * 8 + max_qubits * max_qubits  # gate_count, depth, gate_encodings, adjacency
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Reward weights
        self.alpha = 1.0  # Gate count reduction weight
        self.beta = 0.5   # Depth reduction weight
        self.gamma = 10.0  # Fidelity penalty weight
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_circuit = copy.deepcopy(self.original_circuit)
        self.step_count = 0
        self.last_gate_count = self.current_circuit.size()
        self.last_depth = self.current_circuit.depth()
        
        observation = self._encode_state()
        info = {
            'gate_count': self.last_gate_count,
            'depth': self.last_depth,
            'fidelity': 1.0
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action tuple [action_type, gate_index, param]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action_type, gate_index, param = action
        
        # Apply action
        reward = 0.0
        action_success = False
        
        try:
            if action_type == self.ACTION_REMOVE_GATE:
                action_success = self._remove_gate(gate_index)
            elif action_type == self.ACTION_MERGE_GATES:
                action_success = self._merge_gates(gate_index, param)
            elif action_type == self.ACTION_REPLACE_GATE:
                action_success = self._replace_gate(gate_index, param)
            elif action_type == self.ACTION_COMMUTE_GATES:
                action_success = self._commute_gates(gate_index, param)
            else:  # ACTION_NO_OP
                action_success = True
        except Exception as e:
            # Action failed, small penalty
            reward = -0.1
        
        # Compute metrics
        current_gate_count = self.current_circuit.size()
        current_depth = self.current_circuit.depth()
        
        # Calculate fidelity
        try:
            current_statevector = Statevector.from_instruction(self.current_circuit)
            fidelity = abs(self.original_statevector.inner(current_statevector)) ** 2
        except Exception:
            fidelity = 0.0
        
        # Calculate reward
        delta_g = self.last_gate_count - current_gate_count
        delta_d = self.last_depth - current_depth
        
        reward = (
            self.alpha * delta_g +
            self.beta * delta_d -
            self.gamma * (1.0 - fidelity)
        )
        
        # Update state
        self.last_gate_count = current_gate_count
        self.last_depth = current_depth
        self.step_count += 1
        
        # Check termination conditions
        terminated = fidelity < self.fidelity_threshold
        truncated = self.step_count >= self.max_steps
        
        observation = self._encode_state()
        info = {
            'gate_count': current_gate_count,
            'depth': current_depth,
            'fidelity': fidelity,
            'action_success': action_success,
            'delta_g': delta_g,
            'delta_d': delta_d
        }
        
        return observation, reward, terminated, truncated, info
    
    def _encode_state(self):
        """
        Encode the current circuit state into a feature vector.
        
        Returns:
            np.array: Encoded state vector
        """
        max_gates = 100
        max_qubits = 10
        
        try:
            # Normalize gate count and depth
            gate_count = max(self.current_circuit.size(), 0)
            depth = max(self.current_circuit.depth(), 0)
            gate_count_norm = min(gate_count / max_gates, 1.0) if max_gates > 0 else 0.0
            depth_norm = min(depth / max_gates, 1.0) if max_gates > 0 else 0.0
            
            # Ensure no NaN
            gate_count_norm = 0.0 if np.isnan(gate_count_norm) else gate_count_norm
            depth_norm = 0.0 if np.isnan(depth_norm) else depth_norm
            
            # Gate type encoding (one-hot for common gates)
            gate_encoding = np.zeros(max_gates * 8, dtype=np.float32)
            gate_types = {
                'h': 0, 'x': 1, 'y': 2, 'z': 3,
                'cx': 4, 'cz': 5, 't': 6, 's': 7
            }
            
            for i, instruction in enumerate(self.current_circuit.data[:max_gates]):
                try:
                    gate_name = instruction.operation.name.lower()
                    gate_idx = gate_types.get(gate_name, 0)
                    if i * 8 + gate_idx < len(gate_encoding):
                        gate_encoding[i * 8 + gate_idx] = 1.0
                except Exception:
                    continue
            
            # Adjacency matrix (simplified - connectivity)
            n_qubits = min(self.current_circuit.num_qubits, max_qubits)
            adjacency = np.zeros(max_qubits * max_qubits, dtype=np.float32)
            
            for instruction in self.current_circuit.data:
                try:
                    qubits = [q.index for q in instruction.qubits]
                    if len(qubits) == 2:
                        q1, q2 = min(qubits[0], max_qubits-1), min(qubits[1], max_qubits-1)
                        idx = q1 * max_qubits + q2
                        if 0 <= idx < len(adjacency):
                            adjacency[idx] = 1.0
                except Exception:
                    continue
            
            # Combine all features
            state = np.concatenate([
                [gate_count_norm, depth_norm],
                gate_encoding,
                adjacency
            ]).astype(np.float32)
            
            # Final validation: replace any NaN or inf with 0
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clamp to reasonable range
            state = np.clip(state, -1.0, 1.0)
            
            return state
        except Exception as e:
            # Fallback: return zero state
            print(f"Warning: Error encoding state: {e}, using zero state")
            state_dim = 2 + max_gates * 8 + max_qubits * max_qubits
            return np.zeros(state_dim, dtype=np.float32)
    
    def _remove_gate(self, gate_index):
        """Remove a gate at the specified index."""
        if gate_index >= len(self.current_circuit.data):
            return False
        
        # Create new circuit without the gate
        new_circuit = QuantumCircuit(self.current_circuit.num_qubits)
        for i, instruction in enumerate(self.current_circuit.data):
            if i != gate_index:
                new_circuit.append(instruction.operation, instruction.qubits)
        
        self.current_circuit = new_circuit
        return True
    
    def _merge_gates(self, gate_index, param):
        """Attempt to merge consecutive gates."""
        if gate_index >= len(self.current_circuit.data) - 1:
            return False
        
        # Simple merge: if same gate type on same qubits, remove one
        gate1 = self.current_circuit.data[gate_index]
        gate2 = self.current_circuit.data[gate_index + 1]
        
        if (gate1.operation.name == gate2.operation.name and
            gate1.qubits == gate2.qubits):
            return self._remove_gate(gate_index + 1)
        
        return False
    
    def _replace_gate(self, gate_index, param):
        """Replace a gate with an equivalent one."""
        if gate_index >= len(self.current_circuit.data):
            return False
        
        # Simple replacement: X->Y->Z rotation
        # In practice, this would use gate equivalence rules
        return False  # Placeholder
    
    def _commute_gates(self, gate_index, param):
        """Attempt to commute gates."""
        # Placeholder - would implement gate commutation rules
        return False
    
    def render(self):
        """Render the current circuit state."""
        print(f"Step: {self.step_count}")
        print(f"Gate Count: {self.current_circuit.size()}")
        print(f"Depth: {self.current_circuit.depth()}")
        print(f"Circuit:\n{self.current_circuit}")

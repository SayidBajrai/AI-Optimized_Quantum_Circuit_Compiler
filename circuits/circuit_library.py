"""Library of quantum circuits for training and benchmarking."""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np


class CircuitLibrary:
    """Collection of standard quantum circuits."""
    
    @staticmethod
    def create_ghz(n_qubits=3):
        """
        Create a GHZ (Greenberger-Horne-Zeilinger) state circuit.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            QuantumCircuit: GHZ state circuit
        """
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        return qc
    
    @staticmethod
    def create_bell_state():
        """
        Create a Bell state (maximally entangled 2-qubit state).
        
        Returns:
            QuantumCircuit: Bell state circuit
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc
    
    @staticmethod
    def create_qft(n_qubits=3):
        """
        Create a Quantum Fourier Transform circuit.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            QuantumCircuit: QFT circuit
        """
        qc = QuantumCircuit(n_qubits)
        
        # Apply QFT
        for j in range(n_qubits):
            qc.h(j)
            for k in range(j + 1, n_qubits):
                angle = np.pi / (2 ** (k - j))
                qc.cp(angle, k, j)
        
        # Swap qubits
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - 1 - i)
        
        return qc
    
    @staticmethod
    def create_w_state(n_qubits=3):
        """
        Create a W state circuit.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            QuantumCircuit: W state circuit
        """
        qc = QuantumCircuit(n_qubits)
        
        # Create W state: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        qc.x(0)
        for i in range(1, n_qubits):
            angle = np.arccos(1.0 / np.sqrt(n_qubits - i + 1))
            qc.ry(2 * angle, i)
            for j in range(i):
                qc.cx(i, j)
        
        return qc
    
    @staticmethod
    def create_adder(n_qubits=3):
        """
        Create a quantum adder circuit.
        
        Args:
            n_qubits: Number of qubits (should be even for adder)
            
        Returns:
            QuantumCircuit: Adder circuit
        """
        qc = QuantumCircuit(n_qubits)
        
        # Simple adder: add qubits 0 to n/2-1 to qubits n/2 to n-1
        half = n_qubits // 2
        for i in range(half):
            qc.cx(i, half + i)
            if i < half - 1:
                qc.ccx(i, half + i, half + i + 1)
        
        return qc
    
    @staticmethod
    def create_swap_test(n_qubits=4):
        """
        Create a swap test circuit for comparing quantum states.
        
        Args:
            n_qubits: Number of qubits (must be odd, includes ancilla)
            
        Returns:
            QuantumCircuit: Swap test circuit
        """
        if n_qubits < 3:
            n_qubits = 3
        
        qc = QuantumCircuit(n_qubits)
        
        # Ancilla qubit is qubit 0
        # First state: qubits 1 to (n-1)/2
        # Second state: qubits (n+1)/2 to n-1
        
        qc.h(0)  # Prepare ancilla in superposition
        
        # Apply controlled swaps
        mid = (n_qubits - 1) // 2
        for i in range(1, mid + 1):
            if mid + i < n_qubits:
                qc.cswap(0, i, mid + i)
        
        qc.h(0)  # Measure ancilla
        
        return qc
    
    @staticmethod
    def create_teleportation():
        """
        Create a quantum teleportation circuit.
        
        Returns:
            QuantumCircuit: Teleportation circuit
        """
        qc = QuantumCircuit(3, 3)
        
        # Create Bell pair between qubits 1 and 2
        qc.h(1)
        qc.cx(1, 2)
        
        # Teleport qubit 0
        qc.cx(0, 1)
        qc.h(0)
        
        # Measure and apply corrections (classical feedback)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.cx(1, 2)
        qc.cz(0, 2)
        
        return qc
    
    @staticmethod
    def create_deutsch_jozsa(n_qubits=3):
        """
        Create a Deutsch-Jozsa algorithm circuit.
        
        Args:
            n_qubits: Number of qubits (includes oracle qubit)
            
        Returns:
            QuantumCircuit: Deutsch-Jozsa circuit
        """
        qc = QuantumCircuit(n_qubits, n_qubits - 1)
        
        # Initialize input qubits in |+⟩
        for i in range(n_qubits - 1):
            qc.h(i)
        
        # Oracle qubit in |1⟩
        qc.x(n_qubits - 1)
        qc.h(n_qubits - 1)
        
        # Oracle (balanced function example)
        for i in range(n_qubits - 1):
            qc.cx(i, n_qubits - 1)
        
        # Apply Hadamard to input qubits
        for i in range(n_qubits - 1):
            qc.h(i)
        
        # Measure
        for i in range(n_qubits - 1):
            qc.measure(i, i)
        
        return qc
    
    @staticmethod
    def get_circuit_metrics(qc):
        """
        Compute metrics for a quantum circuit.
        
        Args:
            qc: QuantumCircuit
            
        Returns:
            dict: Dictionary with gate_count, depth, and final_state
        """
        gate_count = qc.size()
        depth = qc.depth()
        
        # Compute final state - remove measurements and classical operations if present
        # Create a copy without classical bits for statevector computation
        qc_for_state = QuantumCircuit(qc.num_qubits)
        for instruction in qc.data:
            op_name = instruction.operation.name
            # Skip measurements, barriers, and conditional gates that depend on classical bits
            if op_name in ['measure', 'barrier']:
                continue
            # Check if operation has classical bits (conditional gates)
            if hasattr(instruction.operation, 'condition') and instruction.operation.condition is not None:
                continue
            # Add the instruction to the state computation circuit
            try:
                qc_for_state.append(instruction.operation, instruction.qubits)
            except Exception:
                # Skip operations that can't be added (e.g., those requiring classical bits)
                continue
        
        # Compute final state
        try:
            if qc_for_state.size() > 0:
                statevector = Statevector.from_instruction(qc_for_state)
                final_state = statevector.data
            else:
                # If no quantum operations, use initial state
                statevector = Statevector.from_label('0' * qc.num_qubits)
                final_state = statevector.data
        except Exception as e:
            # If statevector computation fails, use zero state as fallback
            try:
                statevector = Statevector.from_label('0' * qc.num_qubits)
                final_state = statevector.data
            except Exception:
                statevector = None
                final_state = None
        
        return {
            'gate_count': gate_count,
            'depth': depth,
            'final_state': final_state,
            'statevector': statevector
        }
    
    @staticmethod
    def get_all_circuits():
        """
        Get all available circuits.
        
        Returns:
            dict: Dictionary mapping circuit names to circuits
        """
        return {
            'ghz_3': CircuitLibrary.create_ghz(3),
            'ghz_4': CircuitLibrary.create_ghz(4),
            'ghz_5': CircuitLibrary.create_ghz(5),
            'bell': CircuitLibrary.create_bell_state(),
            'qft_3': CircuitLibrary.create_qft(3),
            'qft_4': CircuitLibrary.create_qft(4),
            'qft_5': CircuitLibrary.create_qft(5),
            'w_3': CircuitLibrary.create_w_state(3),
            'w_4': CircuitLibrary.create_w_state(4),
            'adder_4': CircuitLibrary.create_adder(4),
            'adder_6': CircuitLibrary.create_adder(6),
            'swap_test_3': CircuitLibrary.create_swap_test(3),
            'swap_test_5': CircuitLibrary.create_swap_test(5),
            'teleportation': CircuitLibrary.create_teleportation(),
            'deutsch_jozsa_3': CircuitLibrary.create_deutsch_jozsa(3),
            'deutsch_jozsa_4': CircuitLibrary.create_deutsch_jozsa(4),
        }
    
    @staticmethod
    def get_circuit_equation(circuit_name):
        """
        Get the mathematical equation for a circuit.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            dict: Dictionary with LaTeX equation and description
        """
        equations = {
            'ghz_3': {
                'equation': r'|\text{GHZ}_3\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)',
                'description': 'GHZ state for 3 qubits - maximally entangled state',
                'formula': r'|\text{GHZ}_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})'
            },
            'ghz_4': {
                'equation': r'|\text{GHZ}_4\rangle = \frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)',
                'description': 'GHZ state for 4 qubits - maximally entangled state',
                'formula': r'|\text{GHZ}_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})'
            },
            'bell': {
                'equation': r'|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)',
                'description': 'Bell state (EPR pair) - maximally entangled 2-qubit state',
                'formula': r'|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = (H \otimes I) \cdot CNOT|00\rangle'
            },
            'qft_3': {
                'equation': r'QFT_3|j\rangle = \frac{1}{\sqrt{8}}\sum_{k=0}^{7} e^{2\pi i jk/8}|k\rangle',
                'description': 'Quantum Fourier Transform for 3 qubits',
                'formula': r'QFT_n|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}|k\rangle'
            },
            'qft_4': {
                'equation': r'QFT_4|j\rangle = \frac{1}{\sqrt{16}}\sum_{k=0}^{15} e^{2\pi i jk/16}|k\rangle',
                'description': 'Quantum Fourier Transform for 4 qubits',
                'formula': r'QFT_n|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}|k\rangle'
            },
            'qft_5': {
                'equation': r'QFT_5|j\rangle = \frac{1}{\sqrt{32}}\sum_{k=0}^{31} e^{2\pi i jk/32}|k\rangle',
                'description': 'Quantum Fourier Transform for 5 qubits',
                'formula': r'QFT_n|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}|k\rangle'
            },
            'ghz_5': {
                'equation': r'|\text{GHZ}_5\rangle = \frac{1}{\sqrt{2}}(|00000\rangle + |11111\rangle)',
                'description': 'GHZ state for 5 qubits - maximally entangled state',
                'formula': r'|\text{GHZ}_n\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes n} + |1\rangle^{\otimes n})'
            },
            'w_3': {
                'equation': r'|W_3\rangle = \frac{1}{\sqrt{3}}(|100\rangle + |010\rangle + |001\rangle)',
                'description': 'W state for 3 qubits - symmetric entangled state',
                'formula': r'|W_n\rangle = \frac{1}{\sqrt{n}}\sum_{i=0}^{n-1} |0\rangle^{\otimes i}|1\rangle|0\rangle^{\otimes (n-i-1)}'
            },
            'w_4': {
                'equation': r'|W_4\rangle = \frac{1}{\sqrt{4}}(|1000\rangle + |0100\rangle + |0010\rangle + |0001\rangle)',
                'description': 'W state for 4 qubits - symmetric entangled state',
                'formula': r'|W_n\rangle = \frac{1}{\sqrt{n}}\sum_{i=0}^{n-1} |0\rangle^{\otimes i}|1\rangle|0\rangle^{\otimes (n-i-1)}'
            },
            'adder_4': {
                'equation': r'U_{\text{add}}|a\rangle|b\rangle = |a\rangle|a+b \bmod 2^n\rangle',
                'description': 'Quantum adder circuit for 4 qubits',
                'formula': r'U_{\text{add}}|a\rangle|b\rangle = |a\rangle|a+b \bmod 2^n\rangle'
            },
            'adder_6': {
                'equation': r'U_{\text{add}}|a\rangle|b\rangle = |a\rangle|a+b \bmod 2^n\rangle',
                'description': 'Quantum adder circuit for 6 qubits',
                'formula': r'U_{\text{add}}|a\rangle|b\rangle = |a\rangle|a+b \bmod 2^n\rangle'
            },
            'swap_test_3': {
                'equation': r'P(|\psi\rangle = |\phi\rangle) = \frac{1}{2}(1 + |\langle\psi|\phi\rangle|^2)',
                'description': 'Swap test circuit for comparing quantum states',
                'formula': r'P(|\psi\rangle = |\phi\rangle) = \frac{1}{2}(1 + |\langle\psi|\phi\rangle|^2)'
            },
            'swap_test_5': {
                'equation': r'P(|\psi\rangle = |\phi\rangle) = \frac{1}{2}(1 + |\langle\psi|\phi\rangle|^2)',
                'description': 'Swap test circuit for comparing quantum states (5 qubits)',
                'formula': r'P(|\psi\rangle = |\phi\rangle) = \frac{1}{2}(1 + |\langle\psi|\phi\rangle|^2)'
            },
            'teleportation': {
                'equation': r'|\psi\rangle \rightarrow |\psi\rangle \text{ (teleported)}',
                'description': 'Quantum teleportation protocol',
                'formula': r'|\psi\rangle \text{ on qubit 0} \rightarrow |\psi\rangle \text{ on qubit 2}'
            },
            'deutsch_jozsa_3': {
                'equation': r'U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle',
                'description': 'Deutsch-Jozsa algorithm for 3 qubits',
                'formula': r'U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle'
            },
            'deutsch_jozsa_4': {
                'equation': r'U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle',
                'description': 'Deutsch-Jozsa algorithm for 4 qubits',
                'formula': r'U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle'
            }
        }
        
        return equations.get(circuit_name, {
            'equation': 'N/A',
            'description': 'No equation available',
            'formula': 'N/A'
        })

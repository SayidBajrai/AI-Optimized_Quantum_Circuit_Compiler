# 📖 User Manual - AI-Optimized Quantum Circuit Compiler

## Table of Contents

1. [Getting Started](#getting-started)
2. [Training Control Page](#training-control-page)
3. [Training Results Page](#training-results-page)
4. [Test Circuit Page](#test-circuit-page)
5. [Circuit Library](#circuit-library)
6. [File Formats](#file-formats)
7. [Troubleshooting](#troubleshooting)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

### Starting the Application

1. **Activate the virtual environment** (if not already active):
   ```bash
   .venv\Scripts\activate.bat
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```
   Or use the start script:
   ```bash
   start.bat
   ```

3. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Navigation

The application has four main pages accessible from the navigation menu:

- **Home** (`/`) - Project overview and introduction
- **Circuits** (`/circuit`) - Browse available benchmark circuits
- **Train** (`/train`) - Configure and start training sessions
- **Results** (`/results`) - View training results and metrics
- **Test** (`/test`) - Test circuits with custom input states

---

## Training Control Page

The Training Control page (`/train`) allows you to configure and start reinforcement learning training sessions.

### Circuit Selection

You have two options for selecting a circuit:

#### Option 1: Use Library Circuit

1. Select the **"Use Library Circuit"** radio button
2. Choose a circuit from the dropdown menu:
   - **GHZ States**: `ghz_3`, `ghz_4`, `ghz_5` (3, 4, or 5 qubits)
   - **Bell State**: `bell` (2 qubits)
   - **Quantum Fourier Transform**: `qft_3`, `qft_4`, `qft_5` (3, 4, or 5 qubits)
   - **W States**: `w_3`, `w_4` (3 or 4 qubits)
   - **Adder Circuits**: `adder_4`, `adder_6` (4 or 6 qubits)
   - **Swap Test**: `swap_test_3`, `swap_test_5` (3 or 5 qubits)
   - **Teleportation**: `teleportation` (3 qubits)
   - **Deutsch-Jozsa**: `deutsch_jozsa_3`, `deutsch_jozsa_4` (3 or 4 qubits)

3. The **Original Circuit** diagram will update automatically when you select a circuit

#### Option 2: Upload Custom Circuit (.qasm)

1. Select the **"Upload Custom Circuit (.qasm)"** radio button
2. Click **"Choose File"** and select a `.qasm` file from your computer
3. Click **"Load Custom Circuit"** to upload and validate the file
4. The circuit diagram and information will be displayed

**Note**: The uploaded file must be a valid OpenQASM 2.0 format file.

### Training Configuration

Configure the following parameters:

- **Number of Episodes**: Total number of training episodes (default: 100)
  - More episodes = longer training but potentially better results
  - Recommended: 100-500 for initial experiments

- **Max Steps per Episode**: Maximum actions per episode (default: 50)
  - Limits how many optimization steps the agent can take
  - Higher values allow more exploration but slower training

- **Fidelity Threshold**: Minimum fidelity to maintain (default: 0.99)
  - Circuits below this threshold are penalized
  - Range: 0.0 to 1.0 (1.0 = perfect fidelity)

- **Learning Rate**: Neural network learning rate (default: 0.0001)
  - Controls how fast the agent learns
  - Lower values = more stable but slower learning
  - Recommended: 0.0001 - 0.001

- **Discount Factor (γ)**: Future reward discount (default: 0.99)
  - How much the agent values future rewards vs immediate rewards
  - Range: 0.0 to 1.0 (1.0 = equal weight to all future rewards)

### Save Directory (Optional)

1. Click the **"Browse..."** button next to "Save Directory (Optional)"
2. Select a directory on your computer where you want to save results
3. The path will be filled in automatically
4. **Leave empty** to use the default directory (`results/`)

**Note**: Results include:
- Training history JSON file
- Optimized circuit QASM file
- Model checkpoints (saved every 50 episodes)

### Starting Training

1. Review all configuration settings
2. Click the **"Start Training"** button
3. The training will begin in the background
4. You can monitor progress in real-time:
   - **Training Status**: Shows current episode and status
   - **Current Circuit**: Displays the circuit being optimized
   - **Training Metrics**: Real-time graphs of:
     - Episode Rewards
     - Gate Counts
     - Circuit Depths
     - Fidelities
   - **Original vs Optimized**: Side-by-side circuit comparison

### During Training

- Training runs asynchronously - you can navigate to other pages
- Metrics update automatically every few seconds
- Checkpoints are saved every 50 episodes
- You can stop training at any time using the **"Stop Training"** button

### Training Complete

When training finishes:
- Results are automatically saved to the specified directory (or `results/`)
- Navigate to the **Results** page to view detailed metrics
- The optimized circuit is available for testing

---

## Training Results Page

The Results page (`/results`) displays detailed information about completed training sessions.

### Viewing Results from Default Directory

1. Select a results file from the dropdown menu
2. Click **"Load Results"**
3. The following information will be displayed:

#### Results Details

- **Session ID**: Unique identifier for the training session
- **Circuit Name**: The circuit that was optimized
- **Original Circuit Metrics**:
  - Gate Count
  - Circuit Depth
- **Final/Optimized Metrics**:
  - Final Gate Count
  - Final Depth
  - Number of Episodes

#### Optimization Results

- **Gate Count Reduction**: Absolute and percentage reduction
- **Depth Reduction**: Absolute and percentage reduction
- **Fidelity**: Final fidelity achieved

#### Circuit Diagrams

- **Original Circuit**: The input circuit diagram
- **Optimized Circuit**: The best optimized circuit found during training

#### Training Metrics Plot

- **View Plot** button displays a comprehensive graph showing:
  - Episode Rewards over time
  - Gate Counts over episodes
  - Circuit Depths over episodes
  - Fidelities over episodes

### Loading Results from File

1. Click **"Load File"** button
2. Select a `.json` results file from your computer
3. The results will be displayed with the same information as above
4. The plot will be generated automatically if training history is available

### Understanding the Results

- **Gate Count Reduction**: How many gates were removed
  - Positive values = improvement
  - Higher is better

- **Depth Reduction**: How many layers were reduced
  - Positive values = improvement
  - Important for reducing execution time

- **Fidelity**: How well the optimized circuit preserves the original functionality
  - 1.0 = perfect preservation
  - Should be ≥ 0.99 for valid optimization

- **Training History**: Shows how the agent improved over time
  - Rewards should generally increase
  - Gate counts and depths should decrease
  - Fidelities should remain high

---

## Test Circuit Page

The Test page (`/test`) allows you to test quantum circuits with custom input states and visualize the results.

### Loading a Circuit

You have two options:

#### Option 1: Select Results File

1. Select **"Option 1: Select Results File"** radio button
2. Choose a results file from the dropdown (from default `results/` directory)
3. Click **"Load from Results"**
4. The optimized circuit (or original if no optimized version exists) will be loaded

#### Option 2: Upload Custom Circuit (.qasm)

1. Select **"Option 2: Upload Custom Circuit (.qasm)"** radio button
2. Click **"Choose File"** and select a `.qasm` file
3. Click **"Load Custom Circuit"**
4. The circuit will be loaded and displayed

### Configuring Input States

After loading a circuit, you'll see input state controls for each qubit.

#### Per-Qubit State Selection

For each qubit, you can choose:

- **|0⟩** (Ground State): Standard computational basis state
- **|1⟩** (Excited State): Flipped qubit state
- **|+⟩** (Superposition): Equal superposition of |0⟩ and |1⟩
  - State: (|0⟩ + |1⟩) / √2
- **|-⟩** (Superposition with phase): Superposition with negative phase
  - State: (|0⟩ - |1⟩) / √2
- **Custom**: Specify custom amplitudes
  - **Real part**: Real component of the |0⟩ amplitude
  - **Imaginary part**: Imaginary component of the |0⟩ amplitude
  - The state is automatically normalized

**Example Custom States**:
- `real=1, imag=0` → |0⟩
- `real=0, imag=0` → |1⟩ (after normalization)
- `real=0.707, imag=0` → |+⟩ (approximately)
- `real=0.5, imag=0.5` → Custom complex state

### Number of Shots

- **Number of Shots**: How many times to measure the circuit (default: 1024)
  - More shots = more accurate statistics but slower
  - Range: 1 to 10,000
  - Recommended: 1024-4096 for good balance

### Running the Test

1. Configure input states for all qubits
2. Set the number of shots
3. Click **"Run Test"**
4. Results will appear below:

#### Test Results

**Measurement Counts**:
- Bar chart showing measurement outcomes
- X-axis: Measurement outcomes (e.g., "000", "001", "010", etc.)
- Y-axis: Count of times each outcome was measured
- Click **"Show/Hide Raw Data"** to see exact counts

**State Vector**:
- Line chart showing the magnitude of each state amplitude
- X-axis: State index (0 to 2^n - 1 for n qubits)
- Y-axis: Magnitude of the state amplitude
- Click **"Show/Hide Raw Data"** to see the full statevector representation

### Understanding Test Results

- **Measurement Counts**: Shows the probability distribution of measurement outcomes
  - Higher bars = more likely outcomes
  - Should match the theoretical probabilities

- **State Vector**: Shows the quantum state after circuit execution
  - Magnitude indicates the probability amplitude
  - Complex phases are not shown (only magnitudes)

---

## Circuit Library

The application includes a library of benchmark quantum circuits:

### Entanglement Circuits

- **GHZ States** (`ghz_3`, `ghz_4`, `ghz_5`): Maximally entangled states
  - Creates superposition of all qubits in |0⟩ or all in |1⟩
  - Used for quantum error correction and communication

- **Bell State** (`bell`): Two-qubit maximally entangled state
  - Foundation of quantum teleportation

- **W States** (`w_3`, `w_4`): Symmetric superposition states
  - Different from GHZ - only one qubit is |1⟩ at a time

### Quantum Algorithms

- **Quantum Fourier Transform** (`qft_3`, `qft_4`, `qft_5`): Quantum version of FFT
  - Used in Shor's algorithm and phase estimation

- **Deutsch-Jozsa** (`deutsch_jozsa_3`, `deutsch_jozsa_4`): Quantum algorithm demonstration
  - Shows quantum advantage over classical algorithms

### Quantum Applications

- **Adder Circuits** (`adder_4`, `adder_6`): Quantum arithmetic
  - Performs addition on quantum registers

- **Swap Test** (`swap_test_3`, `swap_test_5`): Quantum state comparison
  - Measures similarity between quantum states

- **Teleportation** (`teleportation`): Quantum state teleportation protocol
  - Demonstrates quantum communication

---

## File Formats

### QASM Files (.qasm)

The application uses **OpenQASM 2.0** format for quantum circuits.

**Example QASM file**:
```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
```

**Requirements**:
- Must start with `OPENQASM 2.0;`
- Must include `qelib1.inc` for standard gates
- Define quantum register with `qreg q[n];`
- Define classical register with `creg c[n];` (optional, for measurements)

### Results Files (.json)

Training results are saved as JSON files with the following structure:

```json
{
  "session_id": "training_results_20260101_120000",
  "circuit_name": "ghz_3",
  "config": {
    "num_episodes": 100,
    "max_steps": 50,
    "fidelity_threshold": 0.99,
    "learning_rate": 0.0001,
    "gamma": 0.99
  },
  "original_circuit": {
    "gate_count": 12,
    "depth": 8,
    "qasm": "..."
  },
  "optimized_circuit": {
    "gate_count": 10,
    "depth": 7,
    "fidelity": 0.9999,
    "qasm": "..."
  },
  "optimization_results": {
    "gate_count_reduction": 2,
    "depth_reduction": 1,
    "gate_count_reduction_percent": 16.67,
    "depth_reduction_percent": 12.5
  },
  "training_history": {
    "episode_rewards": [...],
    "gate_counts": [...],
    "depths": [...],
    "fidelities": [...]
  }
}
```

---

## Troubleshooting

### Training Issues

**Problem**: Training stops with NaN errors
- **Solution**: Reduce learning rate (try 0.0001 or lower)
- **Solution**: Increase fidelity threshold slightly
- **Solution**: Use a simpler circuit for initial testing

**Problem**: No improvement in gate count
- **Solution**: Increase number of episodes
- **Solution**: Increase max steps per episode
- **Solution**: Check that fidelity threshold isn't too strict

**Problem**: Training is very slow
- **Solution**: Reduce number of episodes for testing
- **Solution**: Use smaller circuits (fewer qubits)
- **Solution**: Reduce max steps per episode

### File Upload Issues

**Problem**: "Invalid QASM file" error
- **Solution**: Ensure file is valid OpenQASM 2.0 format
- **Solution**: Check that file has `.qasm` extension
- **Solution**: Verify file includes `OPENQASM 2.0;` header

**Problem**: Circuit diagram not showing
- **Solution**: Wait a few seconds for diagram generation
- **Solution**: Refresh the page
- **Solution**: Check browser console for errors

### Testing Issues

**Problem**: Test results show all zeros
- **Solution**: Check input state configuration
- **Solution**: Verify circuit was loaded correctly
- **Solution**: Try increasing number of shots

**Problem**: State vector shows unexpected values
- **Solution**: Verify input states are configured correctly
- **Solution**: Check that circuit matches expected behavior
- **Solution**: Review circuit diagram for correctness

### General Issues

**Problem**: Page not loading
- **Solution**: Ensure Flask server is running
- **Solution**: Check that port 5000 is not in use
- **Solution**: Try refreshing the page

**Problem**: Results not saving
- **Solution**: Check write permissions for save directory
- **Solution**: Ensure sufficient disk space
- **Solution**: Verify save directory path is valid

---

## Tips and Best Practices

### Training Optimization

1. **Start Small**: Begin with simple circuits (3-4 qubits) to understand the system
2. **Monitor Fidelity**: Always ensure fidelity stays above threshold
3. **Use Checkpoints**: Training saves checkpoints every 50 episodes - you can resume if needed
4. **Experiment with Parameters**: Different circuits may need different learning rates
5. **Compare Results**: Run multiple training sessions and compare outcomes

### Circuit Design

1. **Valid QASM**: Ensure your custom circuits follow OpenQASM 2.0 standards
2. **Test First**: Test custom circuits before training to verify they work
3. **Start Simple**: Begin with small circuits and gradually increase complexity
4. **Document**: Keep notes on what circuits you're testing and why

### Testing

1. **Use Multiple Input States**: Test with different input states to verify circuit behavior
2. **Compare with Theory**: Verify measurement counts match expected probabilities
3. **Visualize State Vectors**: Use state vector graphs to understand circuit output
4. **Test Optimized Circuits**: Always test optimized circuits to ensure they still work correctly

### Results Analysis

1. **Check Fidelity**: Always verify fidelity is maintained
2. **Review Training History**: Look at training plots to understand learning progress
3. **Compare Metrics**: Compare gate count and depth reductions
4. **Save Important Results**: Keep copies of successful training sessions

### Performance Tips

1. **Use Default Directory**: Unless you need custom locations, use default `results/` directory
2. **Close Unused Tabs**: Multiple browser tabs can slow down the application
3. **Monitor System Resources**: Training can be CPU-intensive
4. **Save Regularly**: Results are auto-saved, but keep backups of important experiments

---

## Additional Resources

- **Qiskit Documentation**: https://qiskit.org/documentation/
- **OpenQASM Specification**: https://github.com/Qiskit/openqasm
- **Reinforcement Learning**: For understanding the RL approach used

---

## Support

For issues, questions, or contributions:
1. Check this manual first
2. Review the troubleshooting section
3. Examine error messages carefully
4. Check that all requirements are met

---

**Last Updated**: January 2026
**Version**: 1.0

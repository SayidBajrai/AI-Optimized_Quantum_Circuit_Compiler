# 📘 AI-Optimized Quantum Circuit Compiler

## Overview

This project presents a **reinforcement learning–based quantum circuit compiler** that automatically optimizes quantum circuits by reducing **gate count** and **circuit depth**, while preserving **quantum state fidelity**.

Unlike traditional rule-based compilers, this system **learns optimization strategies** through interaction with a quantum simulator.

---

## Project Overview

### Goal

Train a **Reinforcement Learning (RL) agent** to automatically **optimize quantum circuits** by:
- Reducing **gate count**
- Reducing **circuit depth**
- Preserving the **same quantum functionality** (up to tolerance)

This mimics a **quantum compiler pass**, but learned instead of rule-based.

---

## Key Features

- **Reinforcement Learning–driven circuit optimization** - Uses REINFORCE policy gradient algorithm
- **Fidelity-preserving gate reduction** - Maintains ≥99% fidelity while reducing gates
- **Custom circuit support** - Upload your own `.qasm` files for training and testing
- **Per-qubit input state specification** - Configure individual qubit states (|0⟩, |1⟩, |+⟩, |-⟩, or custom)
- **Custom OpenAI Gym environment** - Tailored quantum circuit optimization environment
- **Qiskit-based quantum simulation** - Full quantum circuit simulation and analysis
- **Web interface using Flask + Tailwind** - Modern, responsive user interface
- **Real-time training visualization** - Live metrics, circuit diagrams, and progress tracking
- **Comprehensive testing tools** - Test circuits with custom input states and visualize results
- **Flexible results management** - Save results to custom directories and load from files
- **Rich benchmark library** - 15+ pre-built quantum circuits (GHZ, QFT, Bell, W-state, Adder, etc.)

---

## System Architecture

The AI-Optimized Quantum Circuit Compiler uses a reinforcement learning approach to optimize quantum circuits. The system is built with a modular architecture that separates concerns into distinct components.

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface Layer                      │
│              (Flask + Tailwind CSS + JavaScript)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Backend API                        │
│  - RESTful endpoints for training control                   │
│  - Session management                                       │
│  - Results retrieval                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              RL Training Engine (PyTorch)                   │
│  - Policy Gradient Agent                                    │
│  - Neural Network (Policy Network)                          │
│  - Training loop management                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Custom Gymnasium Environment                        │
│  - State encoding (circuit representation)                  │
│  - Action space (gate operations)                           │
│  - Reward calculation                                       │
│  - Termination conditions                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Quantum Circuit Simulator (Qiskit)                  │
│  - Circuit simulation                                       │
│  - Gate count & depth calculation                           │
│  - Fidelity computation                                     │
│  - State vector operations                                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### A. Web Interface (Flask + Tailwind)

**Purpose**
- Upload / select quantum circuits
- Start training
- Visualize before vs after optimization

**Pages**
- `/` – Home + project intro
- `/circuit` – Circuit library browser with diagrams and equations
- `/train` – RL training control with:
  - Library circuit selection or custom `.qasm` upload
  - Custom save directory selection
  - Real-time training metrics and circuit visualization
- `/results` – View training results with:
  - Optimization metrics and comparisons
  - Training plots (rewards, gate counts, depths, fidelities)
  - Circuit diagrams (original vs optimized)
  - Load results from JSON files
- `/test` – Test circuits with:
  - Results file selection or custom `.qasm` upload
  - Per-qubit input state configuration
  - Measurement counts visualization
  - State vector visualization

#### B. Flask Backend

**Responsibilities**
- Serve HTML
- Handle user input
- Call RL training routines
- Store experiment results
- Provide API endpoints for circuit diagrams and metrics

#### C. Reinforcement Learning Agent

**Type**
- Policy Gradient (REINFORCE) agent

**Input**
- Encoded quantum circuit state

**Output**
- Action to apply to circuit (rewrite / gate removal / substitution)

#### D. Custom OpenAI Gym-Style Environment

**Core of the project**

Defines:
- `state` – Encoded circuit representation
- `action` – Circuit transformation operations
- `reward` – Based on gate/depth reduction and fidelity
- `termination condition` – Fidelity threshold or max steps

#### E. Quantum Circuit Engine (Qiskit)

**Tasks**
- Simulate circuits
- Compute:
  - Gate count
  - Circuit depth
  - Final quantum state
- Compare fidelity before & after optimization

---

### Data Flow

#### Training Flow

1. **User selects circuit** → Circuit loaded from library
2. **User starts training** → Flask API creates Trainer instance
3. **Trainer initializes** → Creates environment and agent
4. **Episode loop**:
   - Environment resets with original circuit
   - Agent selects action based on current state
   - Environment applies action and computes reward
   - Agent stores experience
   - Repeat until termination
5. **Policy update** → Agent updates policy using REINFORCE
6. **Metrics logging** → Training history stored
7. **Checkpointing** → Model saved periodically

#### State Encoding

The environment encodes the quantum circuit state as a feature vector:

- **Gate count** (normalized)
- **Circuit depth** (normalized)
- **Gate type encoding** (one-hot for each gate)
- **Connectivity matrix** (qubit-qubit interactions)

#### Action Space

Actions are represented as:
- **Action type**: Remove, Merge, Replace, Commute, No-op
- **Gate index**: Which gate to operate on
- **Parameter**: Additional parameter for the action

#### Reward Function

$$
r = \alpha \cdot \Delta G + \beta \cdot \Delta D - \gamma \cdot (1 - F)
$$

Where:
- $\Delta G$ = reduction in gate count
- $\Delta D$ = reduction in depth
- $F$ = fidelity
- $\alpha, \beta, \gamma$ = tunable weights

### Key Design Decisions

1. **Policy Gradient over DQN**: Chosen for continuous action space and better sample efficiency
2. **Custom Environment**: Provides fine-grained control over circuit transformations
3. **Fidelity-based Termination**: Ensures quantum functionality is preserved
4. **Modular Architecture**: Easy to swap components (e.g., different RL algorithms)
5. **Web Interface**: Enables non-technical users to interact with the system

### Future Enhancements

- Support for more gate types and operations
- Hardware-aware optimization (connectivity constraints)
- Multi-objective optimization (Pareto front)
- Transfer learning across circuits
- Distributed training support

---

## Mathematical Formulation

### Quantum Circuit Representation

A quantum circuit ( C ) is a sequence of gates:

$$
C = G_1 \circ G_2 \circ \dots \circ G_n
$$

Each gate ( G_i ) is a unitary matrix:

$$
G_i \in U(2^k)
$$

Final quantum state:

$$
|\psi_{out}\rangle = C |\psi_{in}\rangle
$$

### Optimization Objective

We want a new circuit ( C' ) such that:

$$
|\langle \psi_{out} | \psi'_{out} \rangle|^2 \ge \epsilon
$$

Where:
- ( \epsilon \approx 0.99 ) (fidelity threshold)
- ( \psi'_{out} = C' |\psi_{in}\rangle )

Given an input quantum circuit ( C ), the system learns an optimized circuit ( C' ) such that:
- Gate count is minimized
- Circuit depth is minimized
- Fidelity ≥ 99%

### Reinforcement Learning Setup

#### State ( s_t )

Encodes:
- Gate sequence
- Gate types
- Connectivity
- Current depth

Example features:
- One-hot gate encoding
- Adjacency matrix
- Normalized depth

#### Action ( a_t )

Examples:
- Remove redundant gate
- Merge consecutive gates
- Replace gate with equivalent
- Commute gates

#### Reward ( r_t )

$$
r = \alpha \cdot \Delta G + \beta \cdot \Delta D - \gamma \cdot (1 - F)
$$

Where:
- ( \Delta G ) = reduction in gate count
- ( \Delta D ) = reduction in depth
- ( F ) = fidelity
- ( \alpha, \beta, \gamma ) = tunable weights

#### Episode Termination

- Fidelity below threshold 
- Max steps reached
- No further improvement

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Windows OS (for .bat scripts)

### Quick Setup

#### Step 1: Run Setup Script

Double-click `setup.bat` or run from command line:

```bash
setup.bat
```

This will:
1. Create a virtual environment (`.venv`)
2. Install all required dependencies
3. Set up the project structure

#### Step 2: Activate Virtual Environment

```bash
.venv\Scripts\activate.bat
```

#### Step 3: Start the Application

Option A: Use the start script
```bash
start.bat
```

Option B: Run manually
```bash
python app.py
```

The Flask application will start on `http://localhost:5000`

### Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat  # Windows
# or
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Testing

To verify the installation works:

```bash
python test_training.py
```

This will run a short training session (10 episodes) to test all components.

### Project Structure

After setup, your project should have:

```
.
├── app.py                 # Flask application entry point
├── requirements.txt       # Python dependencies
├── setup.bat             # Setup script for Windows
├── start.bat             # Start script for Windows
├── test_training.py      # Test script
│
├── circuits/             # Quantum circuit library
│   ├── __init__.py
│   └── circuit_library.py
│
├── environment/          # Gymnasium environment
│   ├── __init__.py
│   └── quantum_env.py
│
├── agent/                # RL agent implementation
│   ├── __init__.py
│   └── policy_gradient.py
│
├── training/             # Training utilities
│   ├── __init__.py
│   └── trainer.py
│
├── utils/                # Utility functions
│   ├── __init__.py
│   └── visualization.py
│
├── templates/            # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   ├── circuit.html
│   ├── train.html
│   ├── results.html
│   └── test.html
│
└── results/              # Training results (generated)
    └── checkpoints/      # Model checkpoints
```

### Troubleshooting

#### Issue: "python: command not found"
- Make sure Python is installed and added to PATH
- Try using `python3` instead of `python`

#### Issue: "pip: command not found"
- Install pip: `python -m ensurepip --upgrade`
- Or download get-pip.py and run it

#### Issue: Import errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

#### Issue: Port 5000 already in use
- Change the port in `app.py`: `app.run(port=5001)`
- Or stop the process using port 5000

### Next Steps

1. Open `http://localhost:5000` in your browser
2. Navigate to the **Circuits** page to see available circuits
3. Go to the **Train** page to start training
4. View results on the **Results** page
5. Test optimized circuits on the **Test** page

### Development

For development, you may want to:

1. Install additional development tools:
   ```bash
   pip install pytest black flake8
   ```

2. Run linting:
   ```bash
   flake8 .
   ```

3. Format code:
   ```bash
   black .
   ```

---

## Results

The trained agent demonstrates:
- Significant gate count reduction
- Reduced circuit depth
- Fidelity preservation across multiple benchmark circuits

---

## Use Cases

- NISQ-era quantum compilation
- Hardware-aware optimization
- Research into learning-based quantum control

---

## Paper References

1. **D Maslov, GW Dueck, DM Miller, C Negrevergne**, _Quantum circuit simplification and level compaction_, IEEE TCAD
2. **M Amy, D Maslov, M Mosca, M Roetteler**, _A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits_ 
3. **John Preskill**, _Quantum Computing in the NISQ Era_
4. **T Fösel, P Tighineanu, T Weiss, F Marquardt**, _Reinforcement Learning with Neural Networks for Quantum Feedback_
5. **Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, Seth Lloyd**, _Quantum Machine Learning_, Nature
6. **Navin Khaneja, Timo Reiss, Cindie Kehlet, Thomas Schulte-Herbrüggen, Steffen J. Glaser**, _Optimal control of coupled spin dynamics_ 

---

## License

This project is open source and available for research and educational purposes.

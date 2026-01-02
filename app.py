"""Flask application for quantum circuit optimization RL system."""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import threading
from datetime import datetime
from training.trainer import Trainer
from circuits.circuit_library import CircuitLibrary
from utils.visualization import plot_training_metrics, plot_circuit_comparison
from qiskit import QuantumCircuit
import qiskit
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-rl-compiler-secret-key'

# Global training state
trainer = None
training_thread = None


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/circuit')
def circuit():
    """Circuit selection page."""
    library = CircuitLibrary()
    circuits = library.get_all_circuits()
    
    circuit_info = {}
    for name, qc in circuits.items():
        metrics = library.get_circuit_metrics(qc)
        equation_info = library.get_circuit_equation(name)
        circuit_info[name] = {
            'gate_count': metrics['gate_count'],
            'depth': metrics['depth'],
            'num_qubits': qc.num_qubits,
            'equation': equation_info['equation'],
            'description': equation_info['description'],
            'formula': equation_info['formula']
        }
    
    return render_template('circuit.html', circuits=circuit_info)


@app.route('/train')
def train():
    """Training control page."""
    library = CircuitLibrary()
    circuits = library.get_all_circuits()
    
    # Get circuit info for the page
    circuit_info = {}
    for name, qc in circuits.items():
        metrics = library.get_circuit_metrics(qc)
        circuit_info[name] = {
            'gate_count': metrics['gate_count'],
            'depth': metrics['depth'],
            'num_qubits': qc.num_qubits
        }
    
    return render_template('train.html', circuits=circuit_info)


@app.route('/results')
def results():
    """Results visualization page."""
    # Get available result files
    results_dir = "results"
    result_files = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.startswith("training_results_") and file.endswith(".json"):
                result_files.append(file)
    
    return render_template('results.html', result_files=result_files)


@app.route('/test')
def test():
    """Test optimized circuit page."""
    return render_template('test.html')


# API Endpoints

@app.route('/api/circuits', methods=['GET'])
def api_get_circuits():
    """Get list of available circuits."""
    library = CircuitLibrary()
    circuits = library.get_all_circuits()
    
    circuit_list = []
    for name, qc in circuits.items():
        metrics = library.get_circuit_metrics(qc)
        circuit_list.append({
            'name': name,
            'gate_count': metrics['gate_count'],
            'depth': metrics['depth'],
            'num_qubits': qc.num_qubits
        })
    
    return jsonify(circuit_list)


@app.route('/api/circuit/upload', methods=['POST'])
def api_upload_circuit():
    """Upload a custom .qasm circuit file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.qasm'):
        return jsonify({'error': 'File must be a .qasm file'}), 400
    
    try:
        # Save uploaded file temporarily
        uploads_dir = os.path.join("static", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(uploads_dir, filename)
        file.save(filepath)
        
        # Load circuit from QASM
        qc = QuantumCircuit.from_qasm_file(filepath)
        
        # Get metrics
        library = CircuitLibrary()
        metrics = library.get_circuit_metrics(qc)
        
        # Generate diagram
        diagrams_dir = os.path.join("static", "circuit_diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        diagram_filename = f"custom_{filename.replace('.qasm', '.png')}"
        diagram_path = os.path.join(diagrams_dir, diagram_filename)
        
        try:
            from qiskit.visualization import circuit_drawer
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Custom Circuit\n\n{str(qc)}', 
                   ha='center', va='center', fontsize=10, family='monospace')
            ax.axis('off')
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'diagram_path': diagram_path,
            'circuit_name': filename.replace('.qasm', ''),
            'num_qubits': qc.num_qubits,
            'gate_count': metrics['gate_count'],
            'depth': metrics['depth']
        })
    except Exception as e:
        return jsonify({'error': f'Error loading circuit: {str(e)}'}), 500


@app.route('/api/circuit/custom/<filename>/diagram', methods=['GET'])
def api_get_custom_circuit_diagram(filename):
    """Get diagram for a custom uploaded circuit."""
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    # Convert .qasm to .png if needed
    if filename.endswith('.qasm'):
        diagram_filename = f"custom_{filename.replace('.qasm', '.png')}"
    else:
        diagram_filename = f"custom_{filename}.png"
    diagram_path = os.path.join(diagrams_dir, diagram_filename)
    
    if os.path.exists(diagram_path):
        return send_file(diagram_path, mimetype='image/png')
    else:
        # Try to generate it on the fly if the file exists in uploads
        uploads_dir = os.path.join("static", "uploads")
        if filename.endswith('.qasm'):
            qasm_filepath = os.path.join(uploads_dir, filename)
        else:
            qasm_filepath = os.path.join(uploads_dir, f"{filename}.qasm")
        
        if os.path.exists(qasm_filepath):
            try:
                # Load circuit and generate diagram
                qc = QuantumCircuit.from_qasm_file(qasm_filepath)
                from qiskit.visualization import circuit_drawer
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
                fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return send_file(diagram_path, mimetype='image/png')
            except Exception as e:
                return jsonify({'error': f'Error generating diagram: {str(e)}'}), 500
        
        return jsonify({'error': 'Diagram not found'}), 404


@app.route('/api/circuit/from-qasm', methods=['POST'])
def api_generate_diagram_from_qasm():
    """Generate circuit diagram from QASM string."""
    data = request.json
    qasm_str = data.get('qasm')
    
    if not qasm_str:
        return jsonify({'error': 'No QASM string provided'}), 400
    
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        
        # Generate diagram
        diagrams_dir = os.path.join("static", "circuit_diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        diagram_filename = f"temp_{hash(qasm_str)}.png"
        diagram_path = os.path.join(diagrams_dir, diagram_filename)
        
        try:
            from qiskit.visualization import circuit_drawer
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Circuit Diagram\n\n{str(qc)}', 
                   ha='center', va='center', fontsize=10, family='monospace')
            ax.axis('off')
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        return send_file(diagram_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Error generating diagram: {str(e)}'}), 500


@app.route('/api/train/start', methods=['POST'])
def api_start_training():
    """Start training."""
    global trainer, training_thread
    
    if trainer and trainer.is_training:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    config = {
        'num_episodes': data.get('num_episodes', 100),
        'max_steps_per_episode': data.get('max_steps', 50),
        'fidelity_threshold': data.get('fidelity_threshold', 0.99),
        'learning_rate': data.get('learning_rate', 1e-3),
        'gamma': data.get('gamma', 0.99),
        'circuit_name': data.get('circuit_name', 'ghz_3'),
        'checkpoint_interval': data.get('checkpoint_interval', 50),
        'log_interval': data.get('log_interval', 10),
        'custom_circuit_file': data.get('custom_circuit_file'),  # Path to uploaded .qasm
        'save_directory': data.get('save_directory')  # Custom save directory
    }
    
    trainer = Trainer(config)
    
    # Override save directory if provided
    if config.get('save_directory'):
        trainer.results_dir = config['save_directory']
        os.makedirs(trainer.results_dir, exist_ok=True)
        os.makedirs(os.path.join(trainer.results_dir, "checkpoints"), exist_ok=True)
    
    def train_async():
        trainer.train()
    
    training_thread = threading.Thread(target=train_async)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'status': 'started',
        'session_id': trainer.session_id,
        'config': config
    })


@app.route('/api/train/stop', methods=['POST'])
def api_stop_training():
    """Stop training."""
    global trainer
    
    if not trainer or not trainer.is_training:
        return jsonify({'error': 'No training in progress'}), 400
    
    trainer.stop_training()
    
    return jsonify({'status': 'stopped'})


@app.route('/api/train/status', methods=['GET'])
def api_training_status():
    """Get training status."""
    global trainer
    
    if not trainer:
        return jsonify({
            'is_training': False,
            'episode': 0
        })
    
    return jsonify({
        'is_training': trainer.is_training,
        'episode': trainer.current_episode,
        'session_id': trainer.session_id
    })


@app.route('/api/train/metrics', methods=['GET'])
def api_get_metrics():
    """Get current training metrics."""
    global trainer
    
    if not trainer or not trainer.agent:
        return jsonify({'error': 'No training data available'}), 404
    
    history = trainer.agent.training_history
    
    # Get latest metrics
    latest = {
        'episode': len(history['episode_rewards']) - 1,
        'reward': history['episode_rewards'][-1] if history['episode_rewards'] else 0,
        'gate_count': history['gate_counts'][-1] if history['gate_counts'] else 0,
        'depth': history['depths'][-1] if history['depths'] else 0,
        'fidelity': history['fidelities'][-1] if history['fidelities'] else 0
    }
    
    # Get best metrics if available
    best_metrics = None
    if trainer.best_metrics:
        best_metrics = trainer.best_metrics
    
    # Get original metrics
    original_metrics = None
    if trainer.current_circuit:
        original_metrics = {
            'gate_count': trainer.current_circuit.size(),
            'depth': trainer.current_circuit.depth()
        }
    
    return jsonify({
        'latest': latest,
        'best': best_metrics,
        'original': original_metrics,
        'history': {
            'rewards': history['episode_rewards'][-100:],  # Last 100 episodes
            'gate_counts': history['gate_counts'][-100:],
            'depths': history['depths'][-100:],
            'fidelities': history['fidelities'][-100:]
        }
    })


@app.route('/api/results/<filename>', methods=['GET'])
def api_get_results(filename):
    """Get results from a specific file."""
    results_path = os.path.join("results", filename)
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return jsonify(results)


@app.route('/api/results/list', methods=['GET'])
def api_list_results():
    """List all available results files."""
    results_dir = "results"
    result_files = []
    
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.startswith("training_results_") and file.endswith(".json"):
                result_files.append(file)
    
    return jsonify(result_files)


@app.route('/api/results/<filename>/plot', methods=['GET'])
def api_get_results_plot(filename):
    """Generate and return training metrics plot."""
    results_path = os.path.join("results", filename)
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    plot_path = os.path.join("results", f"plot_{filename.replace('.json', '.png')}")
    plot_training_metrics(results['training_history'], save_path=plot_path)
    
    return send_file(plot_path, mimetype='image/png')


@app.route('/api/results/plot-from-data', methods=['POST'])
def api_plot_from_data():
    """Generate plot from JSON data sent in request."""
    try:
        results = request.json
        
        if not results or 'training_history' not in results:
            return jsonify({'error': 'Invalid results data'}), 400
        
        # Generate plot
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        plot_path = os.path.join(temp_dir, f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png") 
        plot_training_metrics(results['training_history'], save_path=plot_path)
        
        return send_file(plot_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'Error generating plot: {str(e)}'}), 500


@app.route('/api/results/<filename>/circuit/original', methods=['GET'])
def api_get_results_original_circuit(filename):
    """Get original circuit diagram from results."""
    library = CircuitLibrary()
    results_path = os.path.join("results", filename)
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    circuit_name = results.get('circuit_name', 'ghz_3')
    circuits = library.get_all_circuits()
    
    if circuit_name not in circuits:
        return jsonify({'error': 'Circuit not found'}), 404
    
    qc = circuits[circuit_name]
    
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    session_id = results.get('session_id', 'unknown')
    diagram_path = os.path.join(diagrams_dir, f"results_original_{session_id}.png")
    
    if not os.path.exists(diagram_path):
        try:
            from qiskit.visualization import circuit_drawer
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Original Circuit\n\n{str(qc)}', 
                   ha='center', va='center', fontsize=10, family='monospace')
            ax.axis('off')
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/results/<filename>/circuit/optimized', methods=['GET'])
def api_get_results_optimized_circuit(filename):
    """Get optimized circuit diagram from results."""
    results_path = os.path.join("results", filename)
    
    if not os.path.exists(results_path):
        return jsonify({'error': 'Results file not found'}), 404
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    best_circuit_file = results.get('best_circuit_file')
    if not best_circuit_file:
        # If no best circuit file, return a placeholder or use final circuit
        return jsonify({'error': 'No optimized circuit saved'}), 404
    
    circuit_path = os.path.join("results", best_circuit_file)
    if not os.path.exists(circuit_path):
        return jsonify({'error': 'Optimized circuit file not found'}), 404
    
    # Load circuit from QASM
    from qiskit import QuantumCircuit
    try:
        qc = QuantumCircuit.from_qasm_file(circuit_path)
    except Exception as e:
        return jsonify({'error': f'Error loading circuit: {str(e)}'}), 500
    
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    session_id = results.get('session_id', 'unknown')
    diagram_path = os.path.join(diagrams_dir, f"results_optimized_{session_id}.png")
    
    # Always regenerate to ensure it's up to date
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Optimized Circuit\n\n{str(qc)}', 
               ha='center', va='center', fontsize=10, family='monospace')
        ax.axis('off')
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/circuit/<circuit_name>/diagram', methods=['GET'])
def api_get_circuit_diagram(circuit_name):
    """Generate and return circuit diagram as image."""
    library = CircuitLibrary()
    circuits = library.get_all_circuits()
    
    if circuit_name not in circuits:
        return jsonify({'error': 'Circuit not found'}), 404
    
    qc = circuits[circuit_name]
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    diagram_path = os.path.join(diagrams_dir, f"{circuit_name}.png")
    
    # Generate diagram if it doesn't exist
    if not os.path.exists(diagram_path):
        try:
            from qiskit.visualization import circuit_drawer
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = circuit_drawer(qc, output='mpl', style={'backgroundcolor': '#FFFFFF'})
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            # Fallback: create a simple text representation
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'{circuit_name.upper()}\n\n{str(qc)}', 
                   ha='center', va='center', fontsize=10, family='monospace')
            ax.axis('off')
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/train/circuit/original', methods=['GET'])
def api_get_original_circuit():
    """Get the original circuit diagram."""
    global trainer
    
    if not trainer or not trainer.current_circuit:
        return jsonify({'error': 'No training session active'}), 404
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    diagram_path = os.path.join(diagrams_dir, f"original_{trainer.session_id}.png")
    
    # Generate diagram if it doesn't exist
    if not os.path.exists(diagram_path):
        try:
            from qiskit.visualization import circuit_drawer
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = circuit_drawer(trainer.current_circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            # Fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'Original Circuit\n\n{str(trainer.current_circuit)}', 
                   ha='center', va='center', fontsize=10, family='monospace')
            ax.axis('off')
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/train/circuit/optimized', methods=['GET'])
def api_get_optimized_circuit():
    """Get the best optimized circuit diagram."""
    global trainer
    
    if not trainer or not trainer.best_circuit:
        return jsonify({'error': 'No optimized circuit available yet'}), 404
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    diagram_path = os.path.join(diagrams_dir, f"optimized_{trainer.session_id}.png")
    
    # Generate diagram (always regenerate to show latest)
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = circuit_drawer(trainer.best_circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        # Fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Optimized Circuit\n\n{str(trainer.best_circuit)}', 
               ha='center', va='center', fontsize=10, family='monospace')
        ax.axis('off')
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/train/circuit/current', methods=['GET'])
def api_get_current_circuit():
    """Get the current circuit from environment."""
    global trainer
    
    if not trainer or not trainer.env:
        return jsonify({'error': 'No training session active'}), 404
    
    # Create diagrams directory if it doesn't exist
    diagrams_dir = os.path.join("static", "circuit_diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)
    
    diagram_path = os.path.join(diagrams_dir, f"current_{trainer.session_id}.png")
    
    # Get current circuit from environment
    current_circuit = trainer.env.current_circuit if trainer.env.current_circuit else trainer.current_circuit
    
    # Generate diagram
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = circuit_drawer(current_circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        # Fallback
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Current Circuit\n\n{str(current_circuit)}', 
               ha='center', va='center', fontsize=10, family='monospace')
        ax.axis('off')
        fig.savefig(diagram_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return send_file(diagram_path, mimetype='image/png')


@app.route('/api/test/circuit', methods=['POST'])
def api_test_circuit():
    """Test a single circuit."""
    data = request.json
    results_file = data.get('results_file')
    custom_circuit_file = data.get('custom_circuit_file')
    input_state = data.get('input_state', 'zero')
    num_shots = data.get('num_shots', 1024)
    
    circuit = None
    
    if custom_circuit_file:
        # Test custom circuit
        if not os.path.exists(custom_circuit_file):
            return jsonify({'error': 'Custom circuit file not found'}), 404
        
        circuit = QuantumCircuit.from_qasm_file(custom_circuit_file)
    elif results_file:
        # Load from results file - use optimized if available, otherwise original
        results_path = os.path.join("results", results_file)
        if not os.path.exists(results_path):
            return jsonify({'error': 'Results file not found'}), 404
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Try to load optimized circuit first
        best_circuit_file = results.get('best_circuit_file')
        if best_circuit_file:
            circuit_path = os.path.join("results", best_circuit_file)
            if os.path.exists(circuit_path):
                circuit = QuantumCircuit.from_qasm_file(circuit_path)
        
        # Fallback to original circuit
        if circuit is None:
            library = CircuitLibrary()
            circuits = library.get_all_circuits()
            circuit_name = results.get('circuit_name', 'ghz_3')
            
            if circuit_name not in circuits:
                return jsonify({'error': 'Circuit not found'}), 404
            
            circuit = circuits[circuit_name]
    else:
        return jsonify({'error': 'No circuit source provided'}), 400
    
    # Prepare input state
    from qiskit.quantum_info import Statevector
    import numpy as np
    from qiskit import QuantumCircuit as QC
    
    n_qubits = circuit.num_qubits
    
    # Handle per-qubit input states
    input_states = data.get('input_states')
    if input_states and len(input_states) == n_qubits:
        # Build state from per-qubit specifications
        qubit_states = []
        for i, state_spec in enumerate(input_states):
            state_type = state_spec.get('type', '0')
            if state_type == 'custom':
                real = float(state_spec.get('real', 1.0))
                imag = float(state_spec.get('imag', 0.0))
                # Create custom single-qubit state: |ψ⟩ = α|0⟩ + β|1⟩
                # User provides (real, imag) which we interpret as the amplitude for |0⟩
                alpha = complex(real, imag)
                alpha_norm = abs(alpha)
                if alpha_norm > 1:
                    alpha = alpha / alpha_norm
                    alpha_norm = 1.0
                # Calculate beta such that |α|^2 + |β|^2 = 1
                beta_mag = np.sqrt(max(0, 1 - alpha_norm**2))
                # Create statevector: [α, β]
                state_array = np.array([alpha, beta_mag], dtype=complex)
                qubit_states.append(Statevector(state_array))
            elif state_type == '0':
                qubit_states.append(Statevector.from_label('0'))
            elif state_type == '1':
                qubit_states.append(Statevector.from_label('1'))
            elif state_type == '+':
                qubit_states.append(Statevector.from_label('+'))
            elif state_type == '-':
                qubit_states.append(Statevector.from_label('-'))
            else:
                qubit_states.append(Statevector.from_label('0'))  # Default
        
        # Combine qubit states using tensor product
        input_sv = qubit_states[0]
        for i in range(1, len(qubit_states)):
            input_sv = input_sv.tensor(qubit_states[i])
    else:
        # Fallback to old method for backward compatibility
        input_state = data.get('input_state', 'zero')
        if input_state == 'zero':
            input_sv = Statevector.from_label('0' * n_qubits)
        elif input_state == 'plus':
            input_sv = Statevector.from_label('+' * n_qubits)
        else:  # random
            input_sv = Statevector(np.random.rand(2**n_qubits))
            input_sv = input_sv / np.linalg.norm(input_sv)
    
    # Simulate circuit
    output_sv = input_sv.evolve(circuit)
    
    # Run measurements
    from qiskit import Aer
    from qiskit import execute
    
    backend = Aer.get_backend('qasm_simulator')
    
    # Circuit measurements
    meas_circuit = QuantumCircuit(n_qubits, n_qubits)
    meas_circuit.compose(circuit, inplace=True)
    meas_circuit.measure_all()
    
    job = execute(meas_circuit, backend, shots=num_shots)
    counts = job.result().get_counts()
    
    # Normalize measurement counts: handle cases where Qiskit returns results with spaces
    # For example, '000 000' should become '000' for a 3-qubit circuit
    normalized_counts = {}
    for key, value in counts.items():
        # If key contains spaces, split and take the first part (quantum register)
        # Otherwise, take the first n_qubits characters
        if ' ' in key:
            # Split by space and take the first part (quantum register result)
            normalized_key = key.split()[0]
        else:
            # Take only the first n_qubits characters (in case of concatenated results)
            normalized_key = key[:n_qubits] if len(key) > n_qubits else key
        
        # If the normalized key already exists, add the counts together
        if normalized_key in normalized_counts:
            normalized_counts[normalized_key] += value
        else:
            normalized_counts[normalized_key] = value
    
    # Convert state vector to parseable format
    state_data = output_sv.data
    
    # Convert complex numbers to magnitude
    state_magnitudes = [abs(complex(c)) for c in state_data]
    
    # Also keep string representation for display
    state_str = str(output_sv)
    
    return jsonify({
        'measurement_counts': normalized_counts,
        'state': state_str,
        'state_magnitudes': state_magnitudes
    })


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/checkpoints', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/circuit_diagrams', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

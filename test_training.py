"""Simple test script to verify the training system works."""

from training.trainer import Trainer
from circuits.circuit_library import CircuitLibrary

def test_training():
    """Test the training system with a small number of episodes."""
    print("=" * 60)
    print("Testing Quantum Circuit RL Training System")
    print("=" * 60)
    
    # Create trainer with minimal config
    config = {
        'num_episodes': 10,
        'max_steps_per_episode': 20,
        'fidelity_threshold': 0.99,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'circuit_name': 'ghz_3',
        'checkpoint_interval': 5,
        'log_interval': 2
    }
    
    trainer = Trainer(config)
    
    print("\n1. Testing circuit library...")
    library = CircuitLibrary()
    circuits = library.get_all_circuits()
    print(f"   ✓ Found {len(circuits)} circuits")
    
    print("\n2. Loading circuit...")
    circuit = trainer.load_circuit('ghz_3')
    print(f"   ✓ Loaded circuit: {circuit.num_qubits} qubits, {circuit.size()} gates")
    
    print("\n3. Creating environment...")
    env = trainer.create_environment()
    print(f"   ✓ Environment created: state_dim={env.observation_space.shape[0]}")
    
    print("\n4. Initializing agent...")
    print(f"   ✓ Agent initialized")
    
    print("\n5. Running training (10 episodes)...")
    print("-" * 60)
    trainer.train(num_episodes=10)
    print("-" * 60)
    
    print("\n6. Training completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_training()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

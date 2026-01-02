"""Visualization utilities for training metrics and circuits."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer


def plot_training_metrics(training_history, save_path=None):
    """
    Plot training metrics over episodes.
    
    Args:
        training_history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    episodes = range(len(training_history['episode_rewards']))
    
    # Episode rewards
    axes[0, 0].plot(episodes, training_history['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Gate count over episodes
    axes[0, 1].plot(episodes, training_history['gate_counts'])
    axes[0, 1].set_title('Gate Count vs Episodes')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Gate Count')
    axes[0, 1].grid(True)
    
    # Depth over episodes
    axes[1, 0].plot(episodes, training_history['depths'])
    axes[1, 0].set_title('Circuit Depth vs Episodes')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Depth')
    axes[1, 0].grid(True)
    
    # Fidelity over episodes
    axes[1, 1].plot(episodes, training_history['fidelities'])
    axes[1, 1].axhline(y=0.99, color='r', linestyle='--', label='Threshold (0.99)')
    axes[1, 1].set_title('Fidelity vs Episodes')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Fidelity')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_circuit_comparison(original_circuit, optimized_circuit, save_path=None):
    """
    Create side-by-side comparison of original and optimized circuits.
    
    Args:
        original_circuit: Original QuantumCircuit
        optimized_circuit: Optimized QuantumCircuit
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original circuit
    try:
        original_diagram = circuit_drawer(original_circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        axes[0].imshow(original_diagram)
        axes[0].set_title(f'Original Circuit\nGates: {original_circuit.size()}, Depth: {original_circuit.depth()}')
        axes[0].axis('off')
    except Exception:
        axes[0].text(0.5, 0.5, f'Original Circuit\nGates: {original_circuit.size()}\nDepth: {original_circuit.depth()}',
                    ha='center', va='center', fontsize=12)
        axes[0].axis('off')
    
    # Optimized circuit
    try:
        optimized_diagram = circuit_drawer(optimized_circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        axes[1].imshow(optimized_diagram)
        axes[1].set_title(f'Optimized Circuit\nGates: {optimized_circuit.size()}, Depth: {optimized_circuit.depth()}')
        axes[1].axis('off')
    except Exception:
        axes[1].text(0.5, 0.5, f'Optimized Circuit\nGates: {optimized_circuit.size()}\nDepth: {optimized_circuit.depth()}',
                    ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def save_circuit_image(circuit, save_path, title=""):
    """
    Save a circuit diagram as an image.
    
    Args:
        circuit: QuantumCircuit to visualize
        save_path: Path to save the image
        title: Title for the circuit
    """
    try:
        fig = circuit_drawer(circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
        if title:
            fig.suptitle(title, fontsize=14)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        # Fallback: create a simple text representation
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'{title}\n\n{str(circuit)}', 
               ha='center', va='center', fontsize=10, family='monospace')
        ax.axis('off')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

"""
Visualization for encoding comparison experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_resource_comparison(n_features_range, resources, save_dir='results'):
    """Bar chart comparing circuit resources across encodings."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['n_qubits', 'depth', 'gate_count']
    titles = ['Qubits Required', 'Circuit Depth', 'Total Gate Count']
    colors = {'Angle': '#FF6B6B', 'Dense Angle': '#4ECDC4', 'Amplitude': '#45B7D1'}

    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(n_features_range))
        width = 0.25
        for i, (name, data) in enumerate(resources.items()):
            values = [d[metric] for d in data]
            ax.bar(x + i * width, values, width, label=name,
                   color=colors.get(name, 'gray'), alpha=0.8)

        ax.set_xlabel('Number of Features')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(n_features_range)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path / 'resource_comparison.png', dpi=150)
    plt.close()


def plot_accuracy_comparison(encoding_names, accuracies, dataset_name='',
                              save_dir='results'):
    """Bar chart of classification accuracy per encoding."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']

    bars = ax.bar(encoding_names, accuracies,
                  color=colors[:len(encoding_names)], alpha=0.85,
                  edgecolor='#2C3E50')

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11)

    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Classification Accuracy by Encoding ({dataset_name})')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path / 'accuracy_comparison.png', dpi=150)
    plt.close()


def plot_training_curves(results_dict, save_dir='results'):
    """Plot training cost over iterations for each encoding."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#9B59B6']

    for i, (name, data) in enumerate(results_dict.items()):
        ax.plot(data['cost_history'], '-', color=colors[i % len(colors)],
                linewidth=1.5, alpha=0.8, label=name)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (1 - Accuracy)', fontsize=12)
    ax.set_title('Training Convergence by Encoding Strategy')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'training_curves.png', dpi=150)
    plt.close()

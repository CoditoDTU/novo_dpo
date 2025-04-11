import matplotlib.pyplot as  plt
import pandas as pd
import numpy as np
import os 
import sys
import json
from Bio import SeqIO
from typing import Dict, List, Tuple, Union
import math

#JSON_PATH = "data/raw/trainer_state.json"

JSON_PATH = os.path.join(os.getcwd(), "protgpt2_test_rgd_0", "trainer_state.json")
PLOT_NAME = "novo_protgpt2_0.pdf"

def extract_training_metrics(data: List[Dict[str, float]]) -> Dict[str, List[float]]:
    """
    Extract all available metrics from training data.
    
    Args:
        data: List of dictionaries containing training metrics per step
        
    Returns:
        Dictionary with metric names as keys and lists of values
    """
    metrics = {}
    if not data:
        return metrics
        
    # Get all available metric keys from first entry (excluding 'step')
    metric_keys = [key for key in data[0].keys() if key != 'step']
    
    for key in metric_keys:
        metrics[key] = [entry[key] for entry in data]
    
    # Always include steps separately
    metrics['steps'] = [entry['step'] for entry in data]
    
    return metrics



def plot_training_metrics(
    metrics: Dict[str, List[float]],
    variables_to_plot: List[str],
    plot_title: str = "Training Metrics",
    output_path: Union[str, None] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot specified training metrics against steps.
    
    Args:
        metrics: Dictionary containing metrics (from extract_training_metrics)
        variables_to_plot: List of metric names to plot (e.g., ['reward', 'loss'])
        plot_title: Title for the plot
        output_path: If provided, saves plot to this path (e.g., 'training_plot.png')
        figsize: Figure size (width, height) in inches
    """
    if 'steps' not in metrics:
        raise ValueError("Metrics dictionary must contain 'steps'")
    
    print(output_path)
    plt.figure(figsize=figsize)
    
    # Define styles for different metrics
    line_styles = {
        'reward': {'color': 'b', 'marker': 'o', 'linestyle': '-', 'label': 'Reward'},
        'loss': {'color': 'r', 'marker': 's', 'linestyle': '--', 'label': 'Loss'},
        # Add more default styles as needed
    }
    
    for var in variables_to_plot:
        if var not in metrics:
            raise ValueError(f"Metric '{var}' not found in data")
            
        style = line_styles.get(var, {
            'color': None,  # Auto-color
            'marker': None,
            'linestyle': '-',
            'label': var.capitalize()
        })
        
        plt.plot(metrics['steps'], metrics[var], 
                 label=style['label'],
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'])
    
    plt.title(plot_title, fontsize=14)
    plt.xlabel("Iteration Step", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        print(output_path)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_metrics_grid(
    metrics: Dict[str, List[float]],
    variables_to_plot: List[str],
    plot_title: str = "Training Metrics",
    output_path: Union[str, None] = None,
    figsize: Tuple[int, int] = (12, 8),
    rows: Union[int, None] = None,
    cols: Union[int, None] = None
):
    """
    Plot specified training metrics in a grid of subplots.
    
    Args:
        metrics: Dictionary containing metrics (from extract_training_metrics)
        variables_to_plot: List of metric names to plot (e.g., ['reward', 'loss'])
        plot_title: Title for the plot
        output_path: If provided, saves plot to this path (e.g., 'training_plot.pdf')
        figsize: Figure size (width, height) in inches
        rows: Number of rows in subplot grid (auto-calculated if None)
        cols: Number of columns in subplot grid (auto-calculated if None)
    """
    if 'steps' not in metrics:
        raise ValueError("Metrics dictionary must contain 'steps'")
    
    n_plots = len(variables_to_plot)
    
    # Calculate grid layout if not specified
    if rows is None and cols is None:
        cols = min(2, n_plots)
        rows = math.ceil(n_plots / cols)
    elif rows is None:
        rows = math.ceil(n_plots / cols)
    elif cols is None:
        cols = math.ceil(n_plots / rows)
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(plot_title, fontsize=14)
    
    # Flatten axes array for easy iteration
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Define styles for different metrics
    line_styles = {
        'reward': {'color': 'b', 'marker': 'o', 'linestyle': '-', 'label': 'Reward'},
        'loss': {'color': 'r', 'marker': 's', 'linestyle': '--', 'label': 'Loss'},
    }
    
    for i, var in enumerate(variables_to_plot):
        if var not in metrics:
            raise ValueError(f"Metric '{var}' not found in data")
            
        style = line_styles.get(var, {
            'color': None,
            'marker': None,
            'linestyle': '-',
            'label': var.capitalize()
        })
        
        ax = axes[i]
        ax.plot(metrics['epoch'], metrics[var],
                label=style['label'],
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'])
        
        ax.set_title(var.capitalize())
        ax.set_xlabel("epoch")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', format='pdf')
        plt.close()
    else:
        plt.show()




def main():
    
    with open(JSON_PATH, 'r') as f :
        logs = json.load(f)
    
    log_history = logs['log_history']

    metrics = extract_training_metrics(data = log_history)
    metrics_2_plot = ['reward', "loss",'reward_std']

    plot_training_metrics_grid(metrics = metrics,
                      variables_to_plot = metrics_2_plot,
                      plot_title = "test_plot", 
                      output_path = "data/processed/{PLOT_NAME}")



if __name__ == "__main__":
    main()
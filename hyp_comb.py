import pandas as pd
import numpy as np
import os 
import sys
import json
import random
import itertools


NAME = 'winrate_loss_plot_combinations.csv'
# Parameter combinations
param_combinations = {
    'loss_type': ['sigmoid', 'w_sigmoid'],
    'dataset': ['gh114.csv']  # As strings to maintain exact syntax
    # 'betas': [0.001,0.01,0.1, 1],  # np.linspace(start = 0, stop = 1, num = 5)
    # 'epsilons': np.logspace(np.log10(0.16), np.log10(0.6), num=6) #np.linspace(start = 0.001, stop = 0.5, num = 6)
    }

#print(np.linspace(start = 0.001, stop = 0.5, num = 6))
#print(np.logspace(np.log10(0.001), np.log10(0.4), num=6))

# Create output directory
output_dir = "/home/developer/Projects/novo_dpo/configs"
os.makedirs(output_dir, exist_ok=True)

# Generate all combinations
param_names = list(param_combinations.keys())
param_values = list(param_combinations.values())

total_combinations = 1
for v in param_values:
    total_combinations *= len(v)

combinations = list(itertools.product(*param_values))

print(f"Generating {total_combinations} Hydra config files...")

#raise NotImplementedError
# Convert to DataFrame and save as CSV
df = pd.DataFrame(combinations, columns=param_names)
df.insert(0, 'id', range(1, len(df)+1))  # Add ID column starting from 1

csv_path = os.path.join(output_dir, NAME)
df.to_csv(csv_path, index=False)

print(f"CSV saved to {csv_path}")
raise NotImplementedError
for i, combo in enumerate(product(*param_values)):
    # Format the base template with current combination
    
    
    # Create filename with combination identifier
    filename = f"genes_only_reconstruction_model_{i}.yaml"
    filepath = os.path.join(output_dir, filename)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(formatted_yaml.strip())
    
    # Print progress
    if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
        print(f"Generated {i + 1}/{total_combinations}")

print(f"\nAll config files saved to '{output_dir}/'")
import pandas as pd
import numpy as np
import os 
import sys
import json
import random
import itertools


NAME = 'hyperparameter_combinations_small.csv'
# Parameter combinations
param_combinations = {
    'epochs': [1,2,3,4,5],
    'learning_rate': [1e-6, 1e-5, 1e-4],  # As strings to maintain exact syntax
    'betas': [0.1],
    'epsilons': [0.3, 0.1] #np.linspace(start = 0, stop = 2, num = 5)
    }

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

# Convert to DataFrame and save as CSV
df = pd.DataFrame(combinations, columns=param_names)

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
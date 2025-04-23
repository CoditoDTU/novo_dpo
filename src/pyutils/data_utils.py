import numpy as np
import pandas as pd
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler
import json
import matplotlib.pyplot as plt
import os 

def save_dataset(dictionaries, output_path):

    with open(output_path, "w") as f :
        json.dump(dictionaries,
                  f,
                  indent = 4)
        

def generate_random_aa_sequence(length):
    """Generate a random amino acid sequence of given length."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # 20 standard amino acids
    return ''.join(random.choices(amino_acids, k=length))


def parse_sequences_to_preference_dict(df, max_removals=None):
    """
    Parses sequences from a DataFrame into preference dictionary. Prompt-chosen-rejected
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'Sequence' column
        max_removals (int, optional): Max characters to remove from sequence
        
    Returns:
        dict: Dictionary with 'prompt', 'chosen' and 'rejected' lists
    """
    if 'Sequence' not in df.columns:
        raise ValueError("DataFrame must contain 'Sequence' column")
    
    result = {
        "prompt": [],
        "chosen": [],
        "rejected": []

    }
    
    for seq in df['Sequence']:
        if pd.isna(seq) or not isinstance(seq, str) or len(seq.strip()) == 0:
            continue
            
        # Remove random number of characters from end
        if max_removals is None:
            max_removals = len(seq) - 1
            
        num_removals = random.randint(1, min(max_removals, len(seq)-1))
        prompt_part = seq[:-num_removals] if num_removals > 0 else seq
        completion_part = seq[-num_removals:] if num_removals > 0 else "" # chosen cause its real
        rejected_part = generate_random_aa_sequence(length = len(completion_part))
        
        # Append to lists instead of using indices
        result["prompt"].append(prompt_part)
        result["chosen"].append(completion_part)
        result["rejected"].append(completion_part)
        
    return result

def parse_sequences_to_preference_dict_fixedn(df):
    """
    Parses sequences from a DataFrame into a preference dictionary with fixed prompt length of 3.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Sequence' column

    Returns:
        dict: Dictionary with 'prompt', 'chosen', and 'rejected' lists
    """
    if 'Sequence' not in df.columns:
        raise ValueError("DataFrame must contain 'Sequence' column")

    result = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for seq in df['Sequence']:
        if pd.isna(seq) or not isinstance(seq, str) or len(seq.strip()) < 4:
            # Skip sequences that are too short to have a prompt and completion
            continue

        prompt_part = seq[:3]
        completion_part = seq[3:]  # 'chosen' part
        rejected_part = generate_random_aa_sequence(length=len(completion_part))

        result["prompt"].append(prompt_part)
        result["chosen"].append(completion_part)
        result["rejected"].append(rejected_part)

    return result



# DPO DATASET FUNCTIONS


def filter_rows_by_pattern(df: pd.DataFrame, column_name: str, pattern: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing only rows where the specified column contains the given pattern.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column to check.
    - pattern (str): The substring or regex pattern to search for.

    Returns:
    - pd.DataFrame: Filtered DataFrame with rows containing the pattern in the specified column.
    """
    # Ensure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Filter rows where the column contains the pattern
    filtered_df = df[df[column_name].str.startswith(pattern, na=False)]

    return filtered_df

def construct_pairs(Yvec, epsilon):
    '''
    Constructs list of pairs of indexes
    
    Yvec: list of values in descending order from the preference value dataset [int]
    epsilon: Threshold of minimun difference for the pairs to be selected (0-1)
    '''
    Yvec = np.asarray(Yvec)
    N = len(Yvec)

    # Use broadcasting to compute differences efficiently
    # Only consider upper triangle because Y_i > Y_j for i < j due to descending order
    i_idx, j_idx = np.triu_indices(N, k=1)  # i < j

    # Compute differences only for upper triangle
    diff = Yvec[i_idx] - Yvec[j_idx]

    # Apply epsilon condition
    mask = diff > epsilon

    return np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)

def format_string_pairs(strings, index_pairs, N_characters):
    '''
    returns a dictionary for preferences learning for the DPO trainer
    
    strings: List of sequences to compare 
    index_pairs: List of the indexes to select 
    N_characters: int that selects the first N characters for the prompt
    '''

    if N_characters is None:
        N_characters = 1
        force_prompt = 'M'
    else:
        force_prompt = None

    result = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for i, j in index_pairs:
        chosen_str = strings[i]
        rejected_str = strings[j]
        if force_prompt is not None:
            prompt = force_prompt
            chosen = chosen_str
            rejected = rejected_str

        else:
            prompt = chosen_str[:N_characters]  # or could use rejected_str[:N_characters]
            chosen = chosen_str[N_characters:]
            rejected = rejected_str[N_characters:]

        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)

    return result


def plot_logps(chosen_logps: list[int], rejected_logps: list[int], path: str):
    """
    Plots chosen_logps and rejected_logps against each other and saves the plot to the specified path.

    Parameters:
    - chosen_logps (list of int): List of chosen log probabilities.
    - rejected_logps (list of int): List of rejected log probabilities.
    - path (str): File path to save the plot image.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rejected_logps, chosen_logps, label='Chosen Logps', marker='o')

    # Compute min/max range across all points
    all_values = chosen_logps + rejected_logps
    min_val = min(all_values)
    max_val = max(all_values)
    value_range = max_val - min_val
    margin = value_range * 0.1

    # Define the y = x line range
    line_min = min_val - margin
    line_max = max_val + margin
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', label='y = x')

    # Labeling and aesthetics
    plt.xlabel('Rejected Log Probability')
    plt.ylabel('Chosen Log Probability')
    plt.title('Chosen vs Rejected Log Probabilities')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()


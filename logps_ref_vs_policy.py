import pandas as pd
import numpy as np
import os
import sys
import json
import random
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
from src.pyutils.ml_tools import  reward_hydrophobicity
from src.pyutils.data_utils import *
import matplotlib.pyplot as plt
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from datasets import DatasetDict
from tqdm.auto import tqdm  
import csv
import torch
import gc
 
 
nexp = 1
loss_type = 'w_sigmoid'
METRIC_COLUMN = 'target_reg'
# --- Log Probability Calculation Function ---
 
def calculate_log_probs(
    model_or_path,
    tokenizer,
    sequences,
    device="cuda"
):
    """
    Uses a model to calculate log probabilities for a list of sequences.
 
    Args:
        model_or_path (str or Model): Path to a trained model or a model object.
        tokenizer: The tokenizer.
        sequences (list): A list of sequence strings.
        device (str): The device to run on ('cuda' or 'cpu').
   
    Returns:
        pd.DataFrame with sequences and their calculated log probabilities.
    """
    print(f"\nCalculating logps using model from: {model_or_path if isinstance(model_or_path, str) else 'memory'}")
   
    if isinstance(model_or_path, str):
        model = AutoModelForCausalLM.from_pretrained(model_or_path).to(device)
    else:
        model = model_or_path
       
    model.eval()
 
    # Create a dummy dataset to leverage the DPOTrainer's logp calculation
    num_seqs = len(sequences)
    data_dict = {"prompt": ["M"] * num_seqs, "chosen": sequences, "rejected": ["M"] * num_seqs}
    hf_dataset = Dataset.from_dict(data_dict)
 
    # Use a temporary DPOConfig
    temp_training_args = DPOConfig(
        output_dir="./dpo_temp_logp_calc",
        precompute_ref_log_probs=True,
        report_to='none',
        auto_find_batch_size=True,
        remove_unused_columns=False,
    )
 
    trainer = DPOTrainer(
        model=model,
        args=temp_training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )
    trainer.get_train_dataloader()
   
    logps = trainer.train_dataset['ref_chosen_logps']
   
    if isinstance(model_or_path, str):
        del model # Clean up memory if we loaded a new model
        gc.collect()
        torch.cuda.empty_cache()
 
    return logps
 
def construct_pairs_e(Yvec, epsilon):
    '''
    Constructs list of pairs of indexes and their score differences.
   
    Yvec: list of values in descending order from the preference value dataset [int]
    epsilon: Threshold of minimum difference for the pairs to be selected (0-1)
   
    Returns:
        tuple: (index_pairs, pair_differences)
            - index_pairs: np.array of shape (N, 2) with the indices of the pairs.
            - pair_differences: np.array of shape (N,) with the score differences for each pair.
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
   
    # Get the pairs and the corresponding differences that satisfy the condition
    valid_pairs = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]
 
    return valid_pairs, valid_diffs
 
def format_string_pairs_e(strings, index_pairs, pair_diffs, N_characters):
    '''
    returns a dictionary for preferences learning for the DPO trainer
   
    strings: List of sequences to compare
    index_pairs: List of the indexes to select
    pair_diffs: List of the score differences for each pair
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
        "rejected": [],
        "epsilon": []  # The new key for the difference values
    }
 
    # Iterate through both the pairs and their differences simultaneously
    for (i, j), diff_value in zip(index_pairs, pair_diffs):
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
        result["epsilon"].append(diff_value) # Append the actual difference value
 
    return result
 
def prepare_dataset(df_prep, epsilon_prep):
    """Sorts dataframe, constructs pairs, and creates a Hugging Face Dataset."""
    df_sorted = df_prep.sort_values(by='target_reg', ascending=False)
    y_vec = df_sorted['target_reg'].to_list()
    seqs = df_sorted['sequence'].to_list()
    index_pairs, pair_diffs = construct_pairs_e(Yvec=y_vec, epsilon=epsilon_prep)
    data_dict = format_string_pairs_e(
        strings=seqs, index_pairs=index_pairs, pair_diffs=pair_diffs, N_characters=None
    )
    return Dataset.from_dict(data_dict), len(index_pairs)

 
def main_dpo_training_and_eval():
    """
    Trains DPO models and evaluates them by calculating policy logps.
    """
    # --- Configuration ---
    OUTPUT_NAME = 'logps_test_gh114'
    DATA_PATH = 'gh114.csv'
    BASE_MODEL_NAME = "NorseDrunkenSailor/ProtGPT2-with-pad"
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_logps', f'run_{nexp}_{loss_type}')
 
    # Fixed Hyperparameters
    N_EPOCHS = 1
    BETA = 0.01
    LEARNING_RATE = 1e-5
    EPSILON = 0.01 #75  # Epsilon for constructing preference pairs
 
    # Model and Training Configuration
    LOGGING_STEPS = 2
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    ADAM_DECAY = 0.1
 
    # --- Load Tokenizer and Device ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # --- Load and Prepare Initial Data ---
    print(f"Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(filepath_or_buffer=DATA_PATH)
        #df = df.head(3)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'. Exiting.")
        return
 
    part_cols = ['part_0', 'part_1', 'part_2']
    df['split'] = df[part_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    all_sequences = df['sequence'].unique().tolist()
 
    # --- Step 1: Calculate Reference Logps (Before Training) ---
    ref_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
    reference_logps = calculate_log_probs(ref_model, tokenizer, all_sequences, device)
   
    df_results = pd.DataFrame({
        'sequence': all_sequences,
        'logps_reference': reference_logps
    })
    del ref_model # Clean up
    gc.collect()
    torch.cuda.empty_cache()
 
    print("\nInitial DataFrame with reference logps:")
    print(df_results.head())
 
    # --- Step 2: DPO Training Loop ---
    partitions = [df[df['split'] == i] for i in range(len(part_cols))]
 
    for j, df_train in enumerate(partitions):
        partition_name = f"part_{j}"
        print(f"\n{'='*20} Starting Fold #{partition_name} {'='*20}")
       
        # Load a fresh model for each fold
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
 
        # Prepare datasets
        hf_train_dataset, n_pairs_train = prepare_dataset(df_train, EPSILON)
        if n_pairs_train == 0:
            print("Skipping fold: no training pairs generated.")
            continue
 
        model_output_dir = os.path.join(PARENT_OUTPUT_DIR, f"fold_{partition_name}")
       
 
        config_dict = {
                'output_dir': model_output_dir,
                'logging_steps': LOGGING_STEPS,
                'beta': BETA,
                'learning_rate': LEARNING_RATE,
                'num_train_epochs': N_EPOCHS,
                'adam_beta1': ADAM_BETAS[0],
                'adam_beta2': ADAM_BETAS[1],
                'adam_epsilon': ADAM_EPSILON,
                'weight_decay': ADAM_DECAY,
                'precompute_ref_log_probs': True,
                'report_to': 'none',
                'auto_find_batch_size': True,
                'loss_type': loss_type,
            }
       
        training_args = DPOConfig(**config_dict)
       
        trainer = DPOTrainer(
            model=model, args=training_args, train_dataset=hf_train_dataset, processing_class=tokenizer
        )
       
        print(f"--- Starting DPO Training for {partition_name} ---")
        print('Training with', n_pairs_train, 'training pairs' )
        trainer.train()
        print(f"--- Training complete. Model saved to {model_output_dir} ---")
 
        # --- Step 3: Calculate Policy Logps (After Training) ---
        policy_logps = calculate_log_probs(trainer.model, tokenizer, all_sequences, device)
        df_results[f'logps_policy_fold_{j}'] = policy_logps
 
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
 
    print('\n--- All folds processed. Final Results ---')
    print(df_results.head())
    df_results.to_csv(f"dpo_{loss_type}_results_with_logps_{nexp}_.csv", index=False)
    print(f"\nFinal results saved to 'dpo_{loss_type}results_with_logps{nexp}_.csv'")

    # --- Step 4: Merge Metric In-Memory ---
    print(f"\n--- Merging metric column '{METRIC_COLUMN}' into results ---")
    if METRIC_COLUMN not in df.columns:
        print(f"Error: Metric column '{METRIC_COLUMN}' not in {DATA_PATH}. Aborting plot.")
        return

    df_metric = df[['sequence', METRIC_COLUMN]].drop_duplicates(subset=['sequence'])
    df_final = pd.merge(df_results, df_metric, on='sequence', how='left')

    # Save the final, merged DataFrame
    final_output_path = os.path.join(PARENT_OUTPUT_DIR, f'final_report_{loss_type}_{nexp}.csv')
    df_final.to_csv(final_output_path, index=False)
    print(f"Final merged data saved to: {final_output_path}")
    print("Preview of the final merged DataFrame:")
    print(df_final.head())

    
    # --- Step 5: Generate and Display Plot (MODIFIED) ---
    print("\n--- Generating final plot ---")
    if df_final.empty or METRIC_COLUMN not in df_final.columns:
        print("Skipping plotting because the DataFrame is invalid.")
        return

    # Find all policy logp columns to plot their difference from reference
    policy_logp_cols = [col for col in df_final.columns if 'logps_policy_fold' in col]
    
    if 'logps_reference' not in df_final.columns:
        print("Skipping plot: 'logps_reference' column is needed to calculate the difference.")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels = [f'Policy Fold {i}' for i in range(len(policy_logp_cols))]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the difference for each policy fold against the metric value
    for i, policy_col in enumerate(policy_logp_cols):
        if policy_col in df_final:
            # Calculate the difference: policy_logp - reference_logp
            logp_difference = df_final[policy_col] - df_final['logps_reference']
            
            ax.scatter(
                df_final[METRIC_COLUMN],  # X-axis: your metric ('target_reg')
                logp_difference,          # Y-axis: the difference in log probabilities
                color=colors[i % len(colors)],
                alpha=0.6,
                s=20,
                label=labels[i]
            )
            
    # Add a horizontal line at y=0 for a clear baseline
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Reference Baseline (No Change)')

    ax.set_title(f'Logp Difference vs. {METRIC_COLUMN} ({loss_type.capitalize()} Loss)')
    ax.set_xlabel(f"Metric Value ({METRIC_COLUMN})")
    ax.set_ylabel("Log Probability Difference (Policy - Reference)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    # Save the plot to a new file to avoid overwriting
    plot_output_path = os.path.join(PARENT_OUTPUT_DIR, f'final_plot_difference_{loss_type}_{nexp}.png')
    plt.savefig(plot_output_path)
    print(f"Plot saved to: {plot_output_path}")


 
if __name__ == '__main__':
   
    main_dpo_training_and_eval()
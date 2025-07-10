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
import argparse


# INPUT VARIABLES:
FILENAME = 'gh114.csv' # CSV for training and testing models
PATH = os.path.join(os.getcwd(), 'data', 'raw', FILENAME) # PATH to file
RESULT_NAME = 'winrate_losses_exp_0.csv' # MODIFIED: New result file name


def calculate_winrate(df):
    """Calculates the winrate (accuracy). A win is when the model prefers the
    sequence with the higher target value."""
    # Condition 1: The model assigns a higher log-probability to the chosen sequence.
    logps_condition = df['chosen_logps'] > df['rejected_logps']
    # Condition 2: The chosen sequence actually has a higher ground-truth score.
    target_condition = df['chosen_target_reg'] > df['rejected_target_reg']
    # Calculate the mean where both conditions are true.
    winrate = (logps_condition & target_condition).mean()
    return winrate


def memory_stats():
    print("memory allocated: ", torch.cuda.memory_allocated()/1024**2)
    print("memory reserved: ", torch.cuda.memory_reserved()/1024**2)

def get_logps(trainer : DPOTrainer):
    
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']

    return ref_chosen_logps, ref_rejected_logps

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



def main():

    # --- 2. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning with an optional dry run.")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='If set, prints experiment configurations without running training.'
    )
    # --- ADDED: New argument for verbose logging ---
    parser.add_argument(
        '--log-plan',
        action='store_true',
        help='If set, prints the experiment plan before running each training step.'
    )
    args = parser.parse_args()
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_loss_winrate_exp', 'run_0')
    if args.dry_run:
        print("--- 🧪 EXECUTING IN DRY RUN MODE 🧪 ---")
        print("No models will be trained. The script will outline the planned experiments.")
        # --- ADDED: Show where main results and summaries will be saved ---
        print(f"\nMain results will be logged to: '{RESULT_NAME}'")
        print(f"Validation summary DataFrames will be saved in: '{os.path.join(PARENT_OUTPUT_DIR, 'validation_summaries')}'")
    elif args.log_plan:
        print("--- 📋 EXECUTING IN VERBOSE MODE 📋 ---")
        print("The plan for each experiment will be printed before it runs.")
        print(f"\nMain results will be logged to: '{RESULT_NAME}'")
        print(f"Validation summary DataFrames will be saved in: '{os.path.join(PARENT_OUTPUT_DIR, 'validation_summaries')}'")


    # 1. Separate DFs for 3-fold cross-validation
    
    df = pd.read_csv(filepath_or_buffer = PATH) 

    df_part0 = df[(df['part_0'] == 1) & (df['part_1'] == 0) & (df['part_2'] == 0)]
    df_part1 = df[(df['part_1'] == 1) & (df['part_0'] == 0) & (df['part_2'] == 0)]
    df_part2 = df[(df['part_2'] == 1) & (df['part_0'] == 0) & (df['part_1'] == 0)]
    dfs = [df_part0, df_part1, df_part2]

    # Load model and tokenizer once
    MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
    TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")

    # 2. Initialize new results file if it doesn't exist
    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            # Use the new, requested column names
            writer.writerow(['loss_type', 'winrate', 'P_train', 'dataset_name'])

    # Load existing results to avoid re-running processed combinations
    processed_combinations = set()
    if os.path.exists(RESULT_NAME):
        try:
            results_df = pd.read_csv(RESULT_NAME)
            # Update the key to check for already processed runs
            processed_combinations = set(zip(
                results_df['dataset_name'], 
                results_df['Loss type'], 
                results_df['P_train']
            ))
        except (pd.errors.EmptyDataError, KeyError):
            processed_combinations = set()

    # --- Cross-validation loop ---
    for i, df_train in enumerate(dfs):
        # Create a single merged validation set from the other two partitions
        val_indices = [j for j in range(len(dfs)) if j != i]
        df_valid_merged = pd.concat([dfs[val_indices[0]], dfs[val_indices[1]]], ignore_index=True).copy()
        
        # This filtered version is used for merging later to get target values
        df_val_merged_filt = df_valid_merged[['sequence', 'target_reg']]
        
        # 3. Load hyperparameters file (now contains dataset_name and loss_type)
        hyp_file = pd.read_csv(filepath_or_buffer='configs/winrate_loss_plot_combinations.csv')
       
        # 4. FIXED HYPERPARAMETERS (previously from iterrows)
        N_EPOCHS = 1 # Formerly row['epochs']
        BETA = 0.01 # Formerly row['betas']
        LEARNING_RATE = 1e-7 # Formerly row['learning_rate']
        EPSILON = 0 # Formerly row['epsilons']  # Epsilon for constructing preference pairs
        LOGGING_STEPS = 2
        ADAM_BETAS = (0.9, 0.999)
        ADAM_EPSILON = 1e-8
        ADAM_DECAY = 0.1
        
        # --- Hyperparameter loop ---
        for _, row in hyp_file.iterrows():
            current_part = f'part_{i}'
            dataset_name = row['dataset']
            loss_type = row['loss_type']

            # Check if this specific combination has already been processed
            if (dataset_name, loss_type, current_part) in processed_combinations:
                print(f"Skipping already processed combination: {dataset_name}, {loss_type}, {current_part}")
                continue

            # --- Prepare Training Data ---
            df_sorted_train = df_train.sort_values(by='target_reg', ascending=False)
            y_train = df_sorted_train['target_reg'].to_list()
            seq_train = df_sorted_train['sequence'].to_list()
            pairs_train, pairs_train_diff  = construct_pairs_e(Yvec=y_train, epsilon=EPSILON)

            train_dict = format_string_pairs_e(strings=seq_train, index_pairs=pairs_train, N_characters=None, pair_diffs = pairs_train_diff)
            hf_train_dataset = Dataset.from_dict(train_dict)

            # --- Prepare UNIFIED Validation Data ---
            df_sorted_valid = df_valid_merged.sort_values(by='target_reg', ascending=False)
            y_valid = df_sorted_valid['target_reg'].to_list()
            seq_valid = df_sorted_valid['sequence'].to_list()
            pairs_valid, pairs_valid_diff = construct_pairs_e(Yvec=y_valid, epsilon=0) # This should change to maximize validation pairs
            valid_dict = format_string_pairs_e(strings=seq_valid, index_pairs=pairs_valid, N_characters=None, pair_diffs = pairs_valid_diff)
            hf_val_dataset = Dataset.from_dict(valid_dict)

            # --- ADDED: Define summary path here so it's available for the dry run printout ---
            summary_filename = f"summary_{dataset_name}_{loss_type}_{current_part}.csv"
            validation_results_dir = os.path.join(PARENT_OUTPUT_DIR, 'validation_summaries')
            summary_filepath = os.path.join(validation_results_dir, summary_filename)

            # --- 4. HANDLE DRY RUN ---
            if args.dry_run or args.log_plan:
                print("\n-------------------------------------------------")
                print(f"▶️ Experiment Plan:")
                print(f"  - Training Fold: {current_part}")
                print(f"  - Dataset Name: {dataset_name}")
                print(f"  - Loss Type: {loss_type}")
                print(f"  - Num Train Pairs (epsilon={EPSILON}): {len(pairs_train)}")
                print(f"  - Num Validation Pairs (epsilon=0): {len(pairs_valid)}")
                print(f"  - Hyperparameters: N_EPOCHS={N_EPOCHS}, BETA={BETA}, LR={LEARNING_RATE}")
                # --- ADDED: Show the specific path for the summary dataframe ---
                print(f"  - Summary DF Name:        {summary_filename}")
                continue # Skip to the next iteration without training
            
            if args.dry_run:
                continue



            # --- Configure DPO Trainer ---
            # Define a unique output directory for this specific model
            model_output_dir = os.path.join(PARENT_OUTPUT_DIR, f"df_{dataset_name}_fold_{current_part}")
            print(f"Model will be saved to: {model_output_dir}")

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
                'loss_type': loss_type
            }

            training_args = DPOConfig(**config_dict)

            trainer = DPOTrainer(
                model=MODEL,
                args=training_args,
                train_dataset=hf_train_dataset,
                processing_class=TOKENIZER # Correct argument is 'tokenizer'
            )

            print(f"Starting training for {dataset_name} ({loss_type}) on {current_part}")
            trainer.train()

            # --- Evaluation on the single, merged validation set ---
            print('Starting validation...')
            val_trainer = DPOTrainer(
                model=trainer.model,
                args=trainer.args,
                train_dataset=hf_val_dataset, # Use the unified validation dataset
                tokenizer=trainer.tokenizer
            )

            chosen_logps, rejected_logps = get_logps(val_trainer)
            chosen_df = pd.DataFrame({'chosen_sequences': valid_dict['chosen'], 'chosen_logps': chosen_logps})
            chosen_df = pd.merge(chosen_df, df_val_merged_filt, left_on='chosen_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])

            rejected_df = pd.DataFrame({'rejected_sequences': valid_dict['rejected'], 'rejected_logps': rejected_logps})
            rejected_df = pd.merge(rejected_df, df_val_merged_filt, left_on='rejected_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])
            
            summary_df = pd.concat([chosen_df, rejected_df], axis=1)

            # --- LINE YOU REQUESTED TO ADD, with logic to create the directory ---
            os.makedirs(validation_results_dir, exist_ok=True)
            summary_df.to_csv(summary_filepath, index=False)

            winrate = calculate_winrate(summary_df)
            print(f'Validation winrate: {winrate:.4f}')


            # --- Write results to the new CSV format ---
            with open(RESULT_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    loss_type, # type of loss
                    float(winrate), # old accuracy 
                    current_part, # training part
                    dataset_name # gh114 
                ])

            # Add this combination to the processed set
            processed_combinations.add((dataset_name, loss_type, current_part))

            del trainer
            #  del model
            gc.collect()
            torch.cuda.empty_cache()
    print('DONE OwO/')
    if args.dry_run or args.log_plan:
         print(f"\n--- ✅ {('Dry run' if args.dry_run else 'Verbose run')} complete. ---")


if __name__ == "__main__":
    main()

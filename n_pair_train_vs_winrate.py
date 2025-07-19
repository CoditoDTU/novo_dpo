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
import matplotlib.pyplot as plt
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm.auto import tqdm
import csv
import torch
import gc
import argparse
import seaborn as sns

def create_combined_scatterplot(file_path, plot_path):
    """
    Reads experiment results and creates a single scatter plot showing all modalities,
    with a single line representing the average winrate across modalities at each step.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Results file not found at '{file_path}'")
        return
    df = pd.read_csv(file_path)
    print("\n--- 📊 Creating Final Combined Plot 📊 ---")
    print("Loaded results data for plotting:")
    print(df.head())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df,
        x='n_train_pairs',
        y='winrate',
        hue='modality',
        alpha=0.7,
        ax=ax,
        palette='viridis'
    )

    mean_line_df = df.groupby('n_train_pairs')['winrate'].mean().reset_index()

    ax.plot(
        mean_line_df['n_train_pairs'],
        mean_line_df['winrate'],
        color='red',
        lw=2.5,
        linestyle='--',
        label='Average Winrate'
    )

    ax.set_title('Winrate vs. Number of Training Pairs', fontsize=18, weight='bold')
    ax.set_xlabel('Number of Training Pairs', fontsize=12)
    ax.set_ylabel('Winrate', fontsize=12)
    
    ax.legend(title='Legend', fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', dpi=300)
    print(f"\n✅ Successfully saved the combined plot as '{plot_path}'")
    plt.show()

def calculate_winrate(df):
    """Calculates the winrate (accuracy)."""
    logps_condition = df['chosen_logps'] > df['rejected_logps']
    target_condition = df['chosen_target_reg'] > df['rejected_target_reg']
    winrate = (logps_condition & target_condition).mean()
    return winrate

def get_logps(trainer : DPOTrainer, validation_dataset: Dataset):
    """Gets the log probabilities for the validation set."""
    trainer.train_dataset = validation_dataset
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']
    return ref_chosen_logps, ref_rejected_logps

def construct_pairs_e(Yvec, epsilon):
    '''Constructs and sorts pairs of indexes by score difference.'''
    Yvec = np.asarray(Yvec)
    N = len(Yvec)
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = Yvec[i_idx] - Yvec[j_idx]
    mask = diff > epsilon
    
    valid_pairs_indices = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]
    combined = sorted(zip(valid_pairs_indices, valid_diffs), key=lambda x: x[1], reverse=True)
    return combined

def format_string_pairs_e(strings, all_pairs_data, indices_to_use, N_characters):
    '''Formats a dictionary for DPO using a subset of pairs.'''
    result = {"prompt": [], "chosen": [], "rejected": []}
    force_prompt = 'M' if N_characters is None else None
    
    for idx in indices_to_use:
        (i, j), _ = all_pairs_data[idx]
        chosen_str = strings[i]
        rejected_str = strings[j]
        
        if force_prompt:
            prompt, chosen, rejected = force_prompt, chosen_str, rejected_str
        else:
            prompt = chosen_str[:N_characters]
            chosen = chosen_str[N_characters:]
            rejected = rejected_str[N_characters:]

        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
    return result

def print_experiment_plan(args, total_pairs, result_name, plot_filename):
    """Prints a summary of the planned experiment."""
    print("\n--- 🧪 Experiment Plan 🧪 ---")
    print(f"  - Run Number: {args.n_run}")
    print(f"  - Input CSV: {args.input_file}")
    print("\n--- Hyperparameters ---")
    print(f"  - DPO Loss Type: {args.loss}")
    print(f"  - Epsilon for pair generation: {args.epsilon}")
    print(f"  - Beta: 0.01")
    print(f"  - Learning Rate: 1e-5")
    print(f"  - Epochs per run: 1")
    print("\n--- Experiment Details ---")
    print(f"  - Modalities: ['IID', 'max_discrepancy']")
    print(f"  - Iteration: Training pairs will go from 2 to {total_pairs} (in steps of 2).")
    print(f"  - Output results will be saved to: {result_name}")
    print(f"  - Final plot will be saved to: {plot_filename}")

def main():
    # --- 1. SETUP & CONFIGURATION ---
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning with iterative pair sampling.")
    parser.add_argument('--input_file', type=str, default='gh114.csv', help='Name of the input CSV file in the data/raw directory.')
    parser.add_argument('--n_run', type=int, required=True, help='The experiment run number (integer).')
    parser.add_argument('--loss', type=str, default='sigmoid', help='Type of loss to train with (e.g., sigmoid, hinge).')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Minimum score difference to create a preference pair.')
    parser.add_argument('--dry-run', action='store_true', help='Prints the experiment plan and exits.')
    parser.add_argument('--show-plan', action='store_true', help='Prints the experiment plan and then executes.')
    args = parser.parse_args()

    # --- PATHS AND FILENAMES ---
    N_RUN = args.n_run
    FILENAME = args.input_file
    DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw', FILENAME)
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_iterative_exp', f'run_{N_RUN}')
    RESULT_NAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_iterative_exp_{N_RUN}_{args.loss}.csv')
    PLOT_FILENAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_iterative_exp_{N_RUN}_{args.loss}_scatterplot_combined.pdf')
    os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True)

    # --- HYPERPARAMETERS ---
    BETA = 0.01
    LEARNING_RATE = 1e-5
    N_EPOCHS = 1
    EPSILON = args.epsilon
    LOSS_TYPE = args.loss
    MODALITIES = ['IID', 'max_discrepancy']

    # --- 2. DATA PREPARATION ---
    print("--- Preparing data ---")
    df = pd.read_csv(filepath_or_buffer=DATA_PATH)
    df_sorted = df.sort_values(by='target_reg', ascending=False).reset_index(drop=True)
    y_scores = df_sorted['target_reg'].to_list()
    sequences = df_sorted['sequence'].to_list()
    all_possible_pairs_data = construct_pairs_e(Yvec=y_scores, epsilon=EPSILON)
    total_pairs = len(all_possible_pairs_data)
    all_indices = list(range(total_pairs))
    print(f"Total valid pairs found (epsilon > {EPSILON}): {total_pairs}")

    if args.show_plan or args.dry_run:
        print_experiment_plan(args, total_pairs, RESULT_NAME, PLOT_FILENAME)
    
    if args.dry_run:
        print("\n--- Dry run complete. Exiting. ---\n")
        sys.exit(0)

    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['modality', 'n_train_pairs', 'winrate', 'loss_type'])

    TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")

    # --- 3. MAIN EXPERIMENT LOOP ---
    for modality in MODALITIES:
        print(f"\n{'='*20}\nSTARTING MODALITY: {modality}\n{'='*20}")
        for n_train in range(2, total_pairs, 2):
            print(f"\n--- Training with n={n_train} pairs (Modality: {modality}) ---")
            
            if modality == 'max_discrepancy':
                train_indices, valid_indices = all_indices[:n_train], all_indices[n_train:]
            else: # IID
                train_indices = random.sample(all_indices, n_train)
                valid_indices = list(set(all_indices) - set(train_indices))
            
            if not valid_indices:
                print("No more pairs for validation. Stopping this modality.")
                break

            train_dict = format_string_pairs_e(sequences, all_possible_pairs_data, train_indices, N_characters=None)
            valid_dict = format_string_pairs_e(sequences, all_possible_pairs_data, valid_indices, N_characters=None)
            hf_train_dataset = Dataset.from_dict(train_dict)
            hf_val_dataset = Dataset.from_dict(valid_dict)

            model = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
            model_output_dir = os.path.join(PARENT_OUTPUT_DIR, f"model_{modality}_n{n_train}")

            training_args = DPOConfig(
                output_dir=model_output_dir, beta=BETA, learning_rate=LEARNING_RATE,
                num_train_epochs=N_EPOCHS, loss_type=LOSS_TYPE, precompute_ref_log_probs=True,
                report_to='none', auto_find_batch_size=True, logging_steps=10,
            )

            trainer = DPOTrainer(
                model=model, args=training_args,
                train_dataset=hf_train_dataset, tokenizer=TOKENIZER
            )
            trainer.train()

            print('Starting validation...')
            chosen_logps, rejected_logps = get_logps(trainer, hf_val_dataset)

            df_val_sequences = pd.DataFrame({'sequence': valid_dict['chosen'] + valid_dict['rejected']}).drop_duplicates()
            df_val_merged = pd.merge(df_val_sequences, df_sorted[['sequence', 'target_reg']], on='sequence', how='left')

            chosen_df = pd.DataFrame({'chosen_sequences': valid_dict['chosen'], 'chosen_logps': chosen_logps})
            chosen_df = pd.merge(chosen_df, df_val_merged, left_on='chosen_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])
            rejected_df = pd.DataFrame({'rejected_sequences': valid_dict['rejected'], 'rejected_logps': rejected_logps})
            rejected_df = pd.merge(rejected_df, df_val_merged, left_on='rejected_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])
            
            summary_df = pd.concat([chosen_df, rejected_df], axis=1).dropna()
            winrate = calculate_winrate(summary_df)
            print(f'--> Validation winrate for n={n_train}: {winrate:.4f}')

            with open(RESULT_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([modality, n_train, float(winrate), LOSS_TYPE])

            del trainer, model, chosen_df, rejected_df, summary_df, hf_train_dataset, hf_val_dataset
            gc.collect()
            torch.cuda.empty_cache()

    print('\nDONE OwO/')

    if os.path.exists(RESULT_NAME):
        create_combined_scatterplot(RESULT_NAME, PLOT_FILENAME)

if __name__ == "__main__":
    main()
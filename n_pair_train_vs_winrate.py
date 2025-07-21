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
 
# --- Plotting and Helper Functions ---
 
def create_combined_scatterplot(file_path, plot_path):
    """Reads experiment results and creates a scatter plot of winrate vs. training pairs."""
    if not os.path.exists(file_path):
        print(f"❌ Error: Results file not found at '{file_path}'")
        return
    df = pd.read_csv(file_path)
    print("\n--- 📊 Creating Final Combined Plot 📊 ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='n_train_pairs', y='winrate', hue='modality', alpha=0.7, ax=ax, palette='viridis')
    mean_line_df = df.groupby('n_train_pairs')['winrate'].mean().reset_index()
    ax.plot(mean_line_df['n_train_pairs'], mean_line_df['winrate'], color='red', lw=2.5, linestyle='--', label='Average Winrate')
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
    model_prefers_chosen = pd.to_numeric(df['chosen_logps']) > pd.to_numeric(df['rejected_logps'])
    target_is_ordered = pd.to_numeric(df['chosen_target_reg']) > pd.to_numeric(df['rejected_target_reg'])
    winrate = model_prefers_chosen[target_is_ordered].mean()
    return winrate
 
def get_logps(trainer: DPOTrainer):
    """
    Gets the pre-computed reference log probabilities from a trainer instance.
    This works because we create a new trainer instance for validation.
    """
    trainer.get_train_dataloader() # This ensures the pre-computation has run
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']
    return ref_chosen_logps, ref_rejected_logps
 
def construct_pairs_e(Yvec, epsilon):
    '''Constructs and sorts pairs of indexes by score difference.'''
    Yvec = np.asarray(Yvec)
    i_idx, j_idx = np.triu_indices(len(Yvec), k=1)
    diff = Yvec[i_idx] - Yvec[j_idx]
    mask = diff > epsilon
    valid_pairs_indices = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]
    combined = sorted(zip(valid_pairs_indices, valid_diffs), key=lambda x: x[1], reverse=True)
    return combined
 
def format_string_pairs_e(strings, all_pairs_data, indices_to_use, N_characters):
    '''Formats a dictionary for DPO, including the score difference (epsilon).'''
    result = {"prompt": [], "chosen": [], "rejected": [], "epsilon": []}
    force_prompt = 'M' if N_characters is None else None
    for idx in indices_to_use:
        (i, j), diff_value = all_pairs_data[idx]
        chosen_str, rejected_str = strings[i], strings[j]
        prompt = force_prompt if force_prompt else chosen_str[:N_characters]
        chosen = chosen_str if force_prompt else chosen_str[N_characters:]
        rejected = rejected_str if force_prompt else rejected_str[N_characters:]
        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
        result["epsilon"].append(diff_value)
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
    print(f"  - Iteration: Training pairs will go from 10 to {total_pairs} (in steps).")
    print(f"  - Output results will be saved to: {result_name}")
    print(f"  - Final plot will be saved to: {plot_filename}")
 
 
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning with iterative pair sampling.")
    parser.add_argument('--input_file', type=str, default='gh114.csv', help='Path to the input CSV file.')
    parser.add_argument('--n_run', type=int, required=True, help='The experiment run number.')
    parser.add_argument('--loss', type=str, default='sigmoid', help='Type of DPO loss.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Minimum score difference for preference pairs.')
    # ✨ RE-INCORPORATED ARGUMENTS
    parser.add_argument('--dry-run', action='store_true', help='Prints the experiment plan and exits.')
    parser.add_argument('--show-plan', action='store_true', help='Prints the experiment plan and then executes.')
    args = parser.parse_args()
 
    # --- Configuration ---
    OUTPUT_NAME = os.path.splitext(args.input_file)[0]
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_iterative_exp',OUTPUT_NAME, {args.loss},f'run_{args.n_run}')
    RESULT_NAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_iterative_exp_{args.n_run}_{args.loss}.csv')
    PLOT_FILENAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_iterative_exp_{args.n_run}_{args.loss}_scatterplot.pdf')
    os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True)
   
    # --- Hyperparameters ---
    BETA = 0.01
    LEARNING_RATE = 1e-5
    N_EPOCHS = 1
    MODALITIES = ['IID', 'max_discrepancy']
 
    # --- Data Preparation ---
    print("--- Preparing data ---")
    df = pd.read_csv(filepath_or_buffer=args.input_file)
    df_sorted = df.sort_values(by='target_reg', ascending=False).reset_index(drop=True)
    y_scores = df_sorted['target_reg'].to_list()
    sequences = df_sorted['sequence'].to_list()
   
    all_possible_pairs = construct_pairs_e(Yvec=y_scores, epsilon=args.epsilon)
    total_pairs = len(all_possible_pairs)
    all_indices = list(range(total_pairs))
    print(f"Total valid pairs found (epsilon > {args.epsilon}): {total_pairs}")
 
    # ✨ RE-INCORPORATED LOGIC FOR DRY RUN AND SHOW PLAN
    if args.show_plan or args.dry_run:
        print_experiment_plan(args, total_pairs, RESULT_NAME, PLOT_FILENAME)
   
    if args.dry_run:
        print("\n--- Dry run complete. Exiting. ---\n")
        sys.exit(0)
 
    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['modality', 'n_train_pairs', 'n_validation_pairs', 'winrate', 'loss_type'])
 
    TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
    if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token
 
    # --- Experiment Loop ---
    for modality in MODALITIES:
        print(f"\n{'='*20}\nSTARTING MODALITY: {modality}\n{'='*20}")
        for n_train in range(10, total_pairs, max(10, total_pairs // 20)):
            print(f"\n--- Training with n={n_train} pairs (Modality: {modality}) ---")
           
            if modality == 'max_discrepancy':
                train_indices, valid_indices = all_indices[:n_train], all_indices[n_train:]
            else: # IID
                train_indices = random.sample(all_indices, n_train)
                valid_indices = list(set(all_indices) - set(train_indices))
           
            if not valid_indices:
                print("No more pairs for validation. Stopping this modality.")
                break
 
            train_dict = format_string_pairs_e(sequences, all_possible_pairs, train_indices, N_characters=None)
            valid_dict = format_string_pairs_e(sequences, all_possible_pairs, valid_indices, N_characters=None)
            hf_train_dataset = Dataset.from_dict(train_dict)
            hf_val_dataset = Dataset.from_dict(valid_dict)
 
            model = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
            model_output_dir = os.path.join(PARENT_OUTPUT_DIR, f"model_{modality}_n{n_train}")
 
            training_args = DPOConfig(
                output_dir=model_output_dir, beta=BETA, learning_rate=LEARNING_RATE,
                num_train_epochs=N_EPOCHS, loss_type=args.loss, precompute_ref_log_probs=True,
                report_to='none', auto_find_batch_size=True, remove_unused_columns=False
            )
 
            trainer = DPOTrainer(model=model, args=training_args, train_dataset=hf_train_dataset, processing_class=TOKENIZER)
            trainer.train()
 
            # --- Validation Step using your proven method ---
            print('Starting validation...')
            val_trainer = DPOTrainer(
                model=trainer.model,
                args=trainer.args,
                train_dataset=hf_val_dataset,
                processing_class=trainer.tokenizer
            )
 
            chosen_logps, rejected_logps = get_logps(val_trainer)
           
            df_val_merged = df_sorted[['sequence', 'target_reg']].drop_duplicates()
 
            chosen_df = pd.DataFrame({'chosen_sequences': valid_dict['chosen'], 'chosen_logps': chosen_logps})
            chosen_df = pd.merge(chosen_df, df_val_merged, left_on='chosen_sequences', right_on='sequence').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])
           
            rejected_df = pd.DataFrame({'rejected_sequences': valid_dict['rejected'], 'rejected_logps': rejected_logps})
            rejected_df = pd.merge(rejected_df, df_val_merged, left_on='rejected_sequences', right_on='sequence').rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])
           
            summary_df = pd.concat([chosen_df, rejected_df], axis=1).dropna()
           
            if not summary_df.empty:
                winrate = calculate_winrate(summary_df)
                n_valid = len(valid_indices) # Get the number of validation pairs
                print(f'--> Validation winrate for n={n_train}| n_valid={n_valid}: {winrate:.4f}')
                with open(RESULT_NAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([modality, n_train, n_valid, float(winrate), args.loss])
            else:
                print("--> No valid pairs in the validation set to calculate winrate.")
 
            # Memory cleanup
            del trainer, model, val_trainer, summary_df
            gc.collect()
            torch.cuda.empty_cache()
 
    print('\nDONE OwO/')
    if os.path.exists(RESULT_NAME):
        create_combined_scatterplot(RESULT_NAME, PLOT_FILENAME)
 
if __name__ == "__main__":
    main()
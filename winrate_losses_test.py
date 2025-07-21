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
# Assuming these are in your project's src directory
# from src.pyutils.ml_tools import  reward_hydrophobicity
# from src.pyutils.data_utils import *
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
# --- ADDED IMPORTS FOR PLOTTING ---
import seaborn as sns
import matplotlib.pyplot as plt

# It's good practice to define functions that might be used elsewhere
# at the top level of the module.

def calculate_winrate(df):
    """Calculates the winrate (accuracy). A win is when the model prefers the
    sequence with the higher target value."""
    # Condition 1: The model assigns a higher log-probability to the chosen sequence.
    logps_condition = df['chosen_logps'] > df['rejected_logps']
    # Condition 2: The chosen sequence actually has a higher ground-truth score.
    target_condition = df['chosen_target_reg'] > df['rejected_target_reg']
    # A "win" is when the model's preference aligns with the ground truth.
    winrate = (logps_condition & target_condition).mean()
    return winrate


def memory_stats():
    """Prints CUDA memory statistics."""
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def get_logps(trainer: DPOTrainer):
    """Extracts reference log probabilities from the trainer's dataset."""
    # Ensure the dataloader is initialized to access the processed dataset
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']
    return ref_chosen_logps, ref_rejected_logps

def construct_pairs_e(Yvec, epsilon):
    '''
    Constructs list of pairs of indexes and their score differences.
    
    Args:
        Yvec (list or np.array): A list of scores, assumed to be sorted in descending order.
        epsilon (float): The minimum score difference threshold for a pair to be included.
    
    Returns:
        tuple: (index_pairs, pair_differences)
            - index_pairs: np.array of shape (N, 2) with the indices of the pairs.
            - pair_differences: np.array of shape (N,) with the score differences for each pair.
    '''
    Yvec = np.asarray(Yvec)
    N = len(Yvec)

    # Efficiently find all pairs (i, j) where i < j
    i_idx, j_idx = np.triu_indices(N, k=1)

    # Calculate the difference Y_i - Y_j for all these pairs
    diff = Yvec[i_idx] - Yvec[j_idx]

    # Create a mask to filter pairs based on the epsilon threshold
    mask = diff > epsilon
    
    # Apply the mask to get the valid pairs and their differences
    valid_pairs = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]

    return valid_pairs, valid_diffs


def format_string_pairs_e(strings, index_pairs, pair_diffs, N_characters):
    '''
    Formats sequence pairs into a dictionary suitable for the DPO trainer.
    
    Args:
        strings (List[str]): The list of all sequences.
        index_pairs (np.array): Array of index pairs (chosen, rejected).
        pair_diffs (np.array): Array of score differences corresponding to the pairs.
        N_characters (int or None): Number of characters for the prompt. If None, uses 'M'.
    
    Returns:
        dict: A dictionary with "prompt", "chosen", "rejected", and "epsilon" lists.
    '''
    result = {"prompt": [], "chosen": [], "rejected": [], "epsilon": []}
    force_prompt = 'M' if N_characters is None else None

    for (i, j), diff_value in zip(index_pairs, pair_diffs):
        chosen_str = strings[i]
        rejected_str = strings[j]
        
        if force_prompt:
            prompt = force_prompt
            chosen = chosen_str
            rejected = rejected_str
        else:
            prompt = chosen_str[:N_characters]
            chosen = chosen_str[N_characters:]
            rejected = rejected_str[N_characters:]

        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
        result["epsilon"].append(diff_value)

    return result

def create_barplot(file_path, plot_path):
    """
    Reads experiment results from a CSV and creates a bar plot of winrates,
    grouped by loss type and dataset.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Results file not found at '{file_path}'")
        return
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print("⚠️ Warning: Results file is empty. No plot will be generated.")
            return
    except pd.errors.EmptyDataError:
        print("⚠️ Warning: Results file is empty. No plot will be generated.")
        return

    print("\n--- 📊 Creating Final Plot 📊 ---")
    print("Loaded results data for plotting:")
    print(df.head())

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    barplot = sns.barplot(
        data=df,
        x='loss_type',
        y='winrate',
        hue='dataset_name',
        errorbar='sd',
        capsize=.05,
        palette='viridis'
    )
    
    # Overlay individual data points for better visibility
    sns.stripplot(
        data=df,
        x='loss_type',
        y='winrate',
        hue='dataset_name',
        palette='dark:red',
        jitter=True,
        dodge=True,
        alpha=0.8,
        linewidth=1,
        edgecolor='black',
        ax=barplot
    )

    plt.title('Mean Winrate by Loss Type and Dataset', fontsize=16, weight='bold')
    plt.xlabel('Loss Type', fontsize=12)
    plt.ylabel('Mean Winrate', fontsize=12)
    
    if not df['winrate'].empty:
        min_val = df['winrate'].min()
        max_val = df['winrate'].max()
        plt.ylim(bottom=min_val * 0.95, top=max_val * 1.05)
    
    handles, labels = barplot.get_legend_handles_labels()
    num_hue_levels = df['dataset_name'].nunique()
    plt.legend(handles[:num_hue_levels], labels[:num_hue_levels], title='Dataset Name', loc='best')

    plt.tight_layout()
    plt.savefig(plot_path, format='pdf', dpi=300)
    print(f"\n✅ Plot successfully saved to '{plot_path}'")
    plt.show()

def main(N_RUN):

    # --- 1. SET UP ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning experiments from a config file.")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='If set, prints the full experiment plan and exits without training.'
    )
    parser.add_argument(
        '--log-plan',
        action='store_true',
        help='If set, prints the plan for each individual experiment before it runs.'
    )
    args = parser.parse_args()

    # --- 2. DEFINE CORE PATHS AND CONFIGS ---
    HYP_FILE = 'configs/winrate_loss_plot_combinations_3.csv'
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_loss_winrate_exp', f'run_{N_RUN}')
    os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True)
    
    RESULT_NAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_losses_exp_run_{N_RUN}.csv')
    PLOT_FILENAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_losses_exp_run_{N_RUN}_barplot.pdf')

    if not os.path.exists(HYP_FILE):
        print(f"❌ Error: Hyperparameter file not found at '{HYP_FILE}'")
        dummy_data = {
            'dataset': ['gh114.csv', 'gh114.csv', 'cm.csv', 'ppat.csv'],
            'loss_type': ['sigmoid', 'ipo', 'sigmoid', 'ipo'],
            'training_fold': ['part_0', 'part_1', 'part_0', 'part_2']
        }
        pd.DataFrame(dummy_data).to_csv(HYP_FILE, index=False)
        print(f"A dummy config file has been created at '{HYP_FILE}'. Please edit it and run again.")
        return
        
    hyp_df = pd.read_csv(HYP_FILE)

    # --- 3. MODIFIED DRY RUN LOGIC ---
    if args.dry_run:
        print("--- 🧪 EXECUTING IN DRY RUN MODE 🧪 ---")
        print("The script will not train any models. It will outline the planned experiments based on your config file.")
        print(f"\n📝 Using Config File: '{HYP_FILE}'")
        print("\n--- Planned Experiment Combinations ---")
        
        planned_runs = []
        for _, row in hyp_df.iterrows():
            dataset_name = row['dataset']
            # This logic mirrors the main loop to predict the epsilon value
            if 'gh114' in dataset_name:
                epsilon = 0.01
            elif 'cm' in dataset_name:
                epsilon = 0.1
            elif 'ppat' in dataset_name:
                epsilon = 1.0
            else:
                epsilon = 0.0
            
            planned_runs.append({
                "dataset": dataset_name,
                "loss": row['loss_type'],
                "epsilon": epsilon,
                "fold": row['training_fold']
            })
            
        # Print a formatted table of the planned runs
        print(f"\n{'Dataset':<20} | {'Loss Type':<15} | {'Epsilon':<10} | {'Training Fold'}")
        print("-" * 68)
        for run in planned_runs:
            print(f"{run['dataset']:<20} | {run['loss']:<15} | {run['epsilon']:<10.2f} | {run['fold']}")

        print("\n--- ✅ Dry run complete. To run these experiments, execute the script without the --dry-run flag. ---")
        return # Exit the main function immediately

    # --- 4. INITIALIZE RESULTS FILE for actual run ---
    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['loss_type', 'winrate', 'training_fold', 'dataset_name'])

    processed_combinations = set()
    try:
        results_df = pd.read_csv(RESULT_NAME)
        if not results_df.empty:
            processed_combinations = set(zip(
                results_df['dataset_name'], 
                results_df['loss_type'], 
                results_df['training_fold']
            ))
    except (pd.errors.EmptyDataError, KeyError):
        processed_combinations = set()

    # --- 5. MAIN EXPERIMENT LOOP ---
    for _, row in hyp_df.iterrows():
        dataset_name = row['dataset']
        loss_type = row['loss_type']
        training_fold = row['training_fold']

        if (dataset_name, loss_type, training_fold) in processed_combinations:
            print(f"⏭️ Skipping already processed combination: {dataset_name}, {loss_type}, {training_fold}")
            continue

        if 'gh114' in dataset_name:
            EPSILON = 0.01
        elif 'cm' in dataset_name:
            EPSILON = 0.1
        elif 'ppat' in dataset_name:
            EPSILON = 1.0
        else:
            EPSILON = 0.0
        
        print(f"\n{'='*60}\n▶️ Starting Experiment: [{dataset_name} | {loss_type} | {training_fold}] with Epsilon={EPSILON}\n{'='*60}")
        
        DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw', dataset_name)
        if not os.path.exists(DATA_PATH):
            print(f"❌ Error: Dataset file not found at '{DATA_PATH}'. Skipping this run.")
            continue
        
        df = pd.read_csv(filepath_or_buffer=DATA_PATH)
        df_part0 = df[df.get('part_0') == 1]
        df_part1 = df[df.get('part_1') == 1]
        df_part2 = df[df.get('part_2') == 1]
        all_dfs = {'part_0': df_part0, 'part_1': df_part1, 'part_2': df_part2}
        
        if training_fold not in all_dfs:
            print(f"❌ Error: Invalid training_fold '{training_fold}'. Must be one of {list(all_dfs.keys())}. Skipping.")
            continue

        df_train = all_dfs[training_fold]
        val_folds = [key for key in all_dfs.keys() if key != training_fold]
        df_valid_merged = pd.concat([all_dfs[key] for key in val_folds], ignore_index=True)

        if df_train.empty or df_valid_merged.empty:
            print(f"❌ Error: Data split resulted in empty train or validation set for fold '{training_fold}'. Skipping.")
            continue

        df_val_merged_filt = df_valid_merged[['sequence', 'target_reg']].copy()

        df_sorted_train = df_train.sort_values(by='target_reg', ascending=False)
        pairs_train, pairs_train_diff = construct_pairs_e(Yvec=df_sorted_train['target_reg'].to_list(), epsilon=EPSILON)
        train_dict = format_string_pairs_e(strings=df_sorted_train['sequence'].to_list(), index_pairs=pairs_train, N_characters=None, pair_diffs=pairs_train_diff)
        hf_train_dataset = Dataset.from_dict(train_dict)

        df_sorted_valid = df_valid_merged.sort_values(by='target_reg', ascending=False)
        pairs_valid, pairs_valid_diff = construct_pairs_e(Yvec=df_sorted_valid['target_reg'].to_list(), epsilon=0)
        valid_dict = format_string_pairs_e(strings=df_sorted_valid['sequence'].to_list(), index_pairs=pairs_valid, N_characters=None, pair_diffs=pairs_valid_diff)
        hf_val_dataset = Dataset.from_dict(valid_dict)

        # Log plan for individual run if requested (dry-run case is now handled above)
        if args.log_plan:
            print("\n------------------- Individual Experiment Plan -------------------")
            print(f"  - Dataset: {dataset_name}, Training Fold: {training_fold}")
            print(f"  - Loss Type: {loss_type}, Epsilon: {EPSILON}")
            print(f"  - Num Train Pairs: {len(pairs_train)}, Num Validation Pairs: {len(pairs_valid)}")
            print("----------------------------------------------------------------\n")

        TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
        MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
        model_output_dir = os.path.join(PARENT_OUTPUT_DIR, "models", f"{dataset_name.replace('.csv','')}_{loss_type}_{training_fold}")

        training_args = DPOConfig(
            output_dir=model_output_dir, num_train_epochs=1, beta=0.01, learning_rate=1e-5,
            loss_type=loss_type, logging_steps=10, precompute_ref_log_probs=True,
            report_to='none', auto_find_batch_size=True,
        )

        trainer = DPOTrainer(model=MODEL, args=training_args, train_dataset=hf_train_dataset, processing_class=TOKENIZER)

        print(f"🚀 Starting training...")
        trainer.train()

        print('Validating model...')
        val_trainer = DPOTrainer(model=trainer.model, args=trainer.args, train_dataset=hf_val_dataset, processing_class=trainer.processing_class)
        chosen_logps, rejected_logps = get_logps(val_trainer)
        
        summary_df = pd.DataFrame({
            'chosen_logps': chosen_logps, 'rejected_logps': rejected_logps,
            'chosen_sequences': valid_dict['chosen'], 'rejected_sequences': valid_dict['rejected']
        })
        summary_df = pd.merge(summary_df, df_val_merged_filt, left_on='chosen_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])
        summary_df = pd.merge(summary_df, df_val_merged_filt, left_on='rejected_sequences', right_on='sequence', how='left').rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])

        validation_results_dir = os.path.join(PARENT_OUTPUT_DIR, 'validation_summaries')
        os.makedirs(validation_results_dir, exist_ok=True)
        summary_filename = f"summary_{dataset_name.replace('.csv','')}_{loss_type}_{training_fold}.csv"
        summary_filepath = os.path.join(validation_results_dir, summary_filename)
        summary_df.to_csv(summary_filepath, index=False)
        print(f"Validation summary saved to: {summary_filepath}")

        winrate = calculate_winrate(summary_df)
        print(f'✅ Validation Winrate: {winrate:.4f}')

        with open(RESULT_NAME, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([loss_type, float(winrate), training_fold, dataset_name])

        processed_combinations.add((dataset_name, loss_type, training_fold))

        del trainer, val_trainer, MODEL, TOKENIZER, summary_df
        gc.collect()
        torch.cuda.empty_cache()

    print('\n--- 🎉 All experiments complete! 🎉 ---')
    
    if not args.dry_run:
        create_barplot(RESULT_NAME, PLOT_FILENAME)

if __name__ == "__main__":
    N_RUN = 8
    main(N_RUN)
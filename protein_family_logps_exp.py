import pandas as pd
import numpy as np
import os
import sys
import json
import random
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
#Assuming these are custom utility functions, they are commented out for standalone execution.
from src.pyutils.ml_tools import  reward_hydrophobicity
from src.pyutils.data_utils import *
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from datasets import DatasetDict
from tqdm.auto import tqdm
import csv
import torch
import gc
import argparse


# Utility functions

def read_fasta_to_dataframe(fasta_file: str) -> pd.DataFrame:
    """
    Reads a FASTA file and parses it into a pandas DataFrame.

    Args:
        fasta_file (str): The path to the FASTA file.

    Returns:
        pd.DataFrame: A DataFrame with 'name' and 'sequence' columns.
    """
    records = SeqIO.parse(fasta_file, "fasta")
    data = [(record.id, str(record.seq)) for record in records]
    family_df = pd.DataFrame(data, columns=['name', 'sequence'])
    print(f"Read {len(family_df)} sequences from {fasta_file}")
    return family_df

def calculate_log_probs(
    model_or_path,
    tokenizer,
    sequences,
    device="cuda"
):
    """
    Uses a model to calculate log probabilities for a list of sequences.
    """
    if isinstance(model_or_path, str):
        print(f"\nCalculating logps using model from: {model_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_or_path).to(device)
    else:
        print("\nCalculating logps using model from memory...")
        model = model_or_path

    model.eval()
    num_seqs = len(sequences)
    data_dict = {"prompt": ["M"] * num_seqs, "chosen": sequences, "rejected": ["M"] * num_seqs}
    hf_dataset = Dataset.from_dict(data_dict)

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
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return logps


def construct_pairs_e(Yvec, epsilon):

    """Constructs list of pairs of indexes and their score differences."""
    Yvec = np.asarray(Yvec)
    N = len(Yvec)
    i_idx, j_idx = np.triu_indices(N, k=1)
    diff = Yvec[i_idx] - Yvec[j_idx]
    mask = diff > epsilon
    valid_pairs = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]
    return valid_pairs, valid_diffs


def format_string_pairs_e(strings, index_pairs, pair_diffs, N_characters):
    """returns a dictionary for preferences learning for the DPO trainer"""
    if N_characters is None:
        N_characters = 1
        force_prompt = 'M'
    else:
        force_prompt = None
    result = {"prompt": [], "chosen": [], "rejected": [], "epsilon": []}
    for (i, j), diff_value in zip(index_pairs, pair_diffs):
        chosen_str, rejected_str = strings[i], strings[j]
        if force_prompt is not None:
            prompt, chosen, rejected = force_prompt, chosen_str, rejected_str
        else:
            prompt = chosen_str[:N_characters]
            chosen, rejected = chosen_str[N_characters:], rejected_str[N_characters:]
        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
        result["epsilon"].append(diff_value)
    return result


def prepare_dataset(df_prep, epsilon_prep):
    """Sorts dataframe, constructs pairs, and creates a Hugging Face Dataset."""
    df_sorted = df_prep.sort_values(by='target_reg', ascending=False)
    y_vec = df_sorted['target_reg'].to_list()
    seqs = df_sorted['sequence'].to_list()
    index_pairs, pair_diffs = construct_pairs_e(Yvec=y_vec, epsilon=epsilon_prep)
    data_dict = format_string_pairs_e(strings=seqs, index_pairs=index_pairs, pair_diffs=pair_diffs, N_characters=None)
    return Dataset.from_dict(data_dict), len(index_pairs)


def plot_kde(df_logps, output_path, title):
    """
    Fits Kernel Density Estimators to logp data and plots the results.
    """
    print("\n--- Generating Kernel Density Estimate (KDE) plot ---")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_logps.columns)))

    # Define the range for the x-axis based on all data
    all_logps = np.concatenate([df_logps[col].dropna().values for col in df_logps.columns])
    x_min, x_max = all_logps.min(), all_logps.max()
    x_grid = np.linspace(x_min, x_max, 1000)[:, np.newaxis]

    for i, col in enumerate(df_logps.columns):
        logps_data = df_logps[col].dropna().values[:, np.newaxis]
        if len(logps_data) == 0:
            print(f"Skipping empty column: {col}")
            continue

        # Fit KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(logps_data)
        log_dens = kde.score_samples(x_grid)
        density = np.exp(log_dens)

        # Plotting
        ax.plot(x_grid, density, color=colors[i], lw=2, label=col)
        ax.fill_between(x_grid.flatten(), density, color=colors[i], alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Log Probability")
    ax.set_ylabel("Density")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"KDE plot saved to: {output_path}")
    plt.close()


def print_experiment_plan(args, parent_dir):
    """Prints a formatted summary of the experiment configuration."""
    header = " 🧪 DRY RUN MODE 🧪 " if args.dry_run else " 📋 EXPERIMENT PLAN 📋 "
    print("\n" + "="*20 + header + "="*20)
    if args.dry_run:
        print("This script will NOT run training or calculations. It will only display the configuration.\n")
    else:
        print("The following experiment will be run:\n")

    # 1. Base model
    print(f"➡️ 1. Base Model: {args.base_model_name}")

    # 2. FASTA file
    print(f"➡️ 2. FASTA File for Analysis: {os.path.abspath(args.fasta_file)}")

    # 3. Train dataset
    print(f"➡️ 3. Training Dataset: {os.path.abspath(args.data_path)}")

    # 4. Hyperparameters
    print("➡️ 4. Hyperparameters:")
    print(f"    - DPO Loss Type: {args.loss_type}")
    print(f"    - Number of Epochs: {args.n_epochs}")
    print(f"    - Beta (β): {args.beta}")
    print(f"    - Learning Rate: {args.learning_rate}")
    print(f"    - Epsilon (ε) for Pair Construction: {args.epsilon}")

    # 5. Output Paths
    final_family_logps_path = os.path.join(parent_dir, f'final_family_logps_{args.loss_type}_{args.nexp}.csv')
    kde_plot_path = os.path.join(parent_dir, f'kde_plot_{args.loss_type}_{args.nexp}.png')
    model_output_dir_example = os.path.join(parent_dir, "fold_part_0")

    print("➡️ 5. Output Paths:")
    print(f"    - Main Output Directory: {os.path.abspath(parent_dir)}")
    print(f"    - Model Checkpoints (example for fold 0): {os.path.abspath(model_output_dir_example)}")
    print(f"    - Final Family Logps CSV: {os.path.abspath(final_family_logps_path)}")
    print(f"    - Final KDE Plot Image: {os.path.abspath(kde_plot_path)}")

    print("\n" + "="*(42 + len(header)) + "\n")



def main_dpo_training_and_eval(args):
    """
    Trains DPO models and evaluates them by calculating policy logps on a target FASTA file.
    """
    # --- Configuration & Plan Display ---
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'gh114_family_logps', f'run_{args.nexp}_{args.loss_type}')

    # If --dry_run or --show_plan is used, display the configuration.
    if args.dry_run or args.show_plan:
        print_experiment_plan(args, PARENT_OUTPUT_DIR)

    # If --dry_run is used, exit after showing the plan.
    if args.dry_run:
        return

    # --- Start of Actual Run ---
    os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True)

    # --- Load Tokenizer and Device ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load and Prepare Datasets ---
    print(f"Loading training data from: {args.data_path}")
    try:
        df = pd.read_csv(filepath_or_buffer=args.data_path)
    except FileNotFoundError:
        print(f"Error: Training data file not found at '{args.data_path}'. Exiting.")
        return

    try:
        family_df = read_fasta_to_dataframe(args.fasta_file)
        family_sequences = family_df['sequence'].tolist()
    except FileNotFoundError:
        print(f"Error: FASTA file not found at '{args.fasta_file}'. Exiting.")
        return

    # Prepare training data partitions
    part_cols = ['part_0', 'part_1', 'part_2']
    df['split'] = df[part_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    partitions = [df[df['split'] == i] for i in range(len(part_cols))]

    # --- Step 2: Calculate Reference Logps for the Family sequences ---
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model_name).to(device)
    family_reference_logps = calculate_log_probs(ref_model, tokenizer, family_sequences, device)
    
    family_logps_df = pd.DataFrame({
        'logps_reference_family': family_reference_logps
    })
    
    del ref_model # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    # --- DPO Training Loop ---
    for j, df_train in enumerate(partitions):
        partition_name = f"part_{j}"
        print(f"\n{'='*20} Starting Fold #{partition_name} {'='*20}")

        model = AutoModelForCausalLM.from_pretrained(args.base_model_name).to(device)
        hf_train_dataset, n_pairs_train = prepare_dataset(df_train, args.epsilon)
        if n_pairs_train == 0:
            print("Skipping fold: no training pairs generated.")
            continue

        model_output_dir = os.path.join(PARENT_OUTPUT_DIR, f"fold_{partition_name}")
        training_args = DPOConfig(
            output_dir=model_output_dir,
            num_train_epochs=args.n_epochs,
            beta=args.beta,
            learning_rate=args.learning_rate,
            loss_type=args.loss_type,
            logging_steps=2,
            precompute_ref_log_probs=True,
            report_to='none',
            auto_find_batch_size=True,
        )
        trainer = DPOTrainer(
            model=model, args=training_args, train_dataset=hf_train_dataset,
            # Per user request, this line is not modified
            processing_class=tokenizer
        )
        print(f"--- Starting DPO Training for {partition_name} with {n_pairs_train} pairs ---")
        trainer.train()
        print(f"--- Training complete. Model saved to {model_output_dir} ---")

        # --- Step 3: Calculate Policy Logps for the Family sequences ---
        trained_partition_name_family_logps = calculate_log_probs(trainer.model, tokenizer, family_sequences, device)
        family_logps_df[f'logps_policy_family_fold_{j}'] = trained_partition_name_family_logps
        
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    # --- Final Analysis and Plotting ---
    print('\n--- All folds processed. Final Family Logps ---')
    print(family_logps_df.head())
    
    final_family_logps_path = os.path.join(PARENT_OUTPUT_DIR, f'final_family_logps_{args.loss_type}_{args.nexp}.csv')
    family_logps_df.to_csv(final_family_logps_path, index=False)
    print(f"Final family logps saved to: {final_family_logps_path}")

    # --- Step 4: Generate KDE Plot ---
    kde_plot_path = os.path.join(PARENT_OUTPUT_DIR, f'kde_plot_{args.loss_type}_{args.nexp}.png')
    plot_kde(
        df_logps=family_logps_df,
        output_path=kde_plot_path,
        title=f'KDE of Family Log Probabilities (Loss: {args.loss_type})'
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DPO training and evaluate log probabilities on a FASTA file.")
    
    # --- File Path Arguments ---
    parser.add_argument('--data_path', type=str, default='gh114.csv', help='Path to the training data CSV file.')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the target FASTA file for analysis.')
    
    # --- Model and Training Arguments ---
    parser.add_argument('--base_model_name', type=str, default="NorseDrunkenSailor/ProtGPT2-with-pad", help="Base model for DPO training.")
    parser.add_argument('--nexp', type=int, default=1, help='Experiment number for output naming.')
    parser.add_argument('--loss_type', type=str, default='sigmoid', choices=[
                    "sigmoid",
                   "w_sigmoid",
                   'w_sigmoid_2',
                   "hinge",
                   "ipo",
                    "exo_pair",
                    "nca_pair",
                    "robust",
                    "bco_pair",
                    "sppo_hard",
                    "aot",
                    "aot_pair",
                    "discopop",
                    "apo_zero",
                    "apo_down"], help='DPO loss function type.')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--beta', type=float, default=0.01, help='Beta parameter for DPO.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon for constructing preference pairs.')
    
    # --- Execution Control Arguments ---
    parser.add_argument('--dry_run', action='store_true', help='If set, prints the experiment configuration and exits without running.')
    parser.add_argument('--show_plan', action='store_true', help='Prints the experiment configuration before running the script.')

    args = parser.parse_args()
    main_dpo_training_and_eval(args)
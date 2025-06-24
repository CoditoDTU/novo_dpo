#%%
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
#%%
#DPO framework2: where we are not grid searching anymore but running validation + custom hf forked implementation


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


#%%

FILENAME = 'gh114.csv' # CSV for traniing and testing models
PATH = os.path.join(os.getcwd(), 'data','raw', FILENAME) # PATH to file
df_train = pd.read_csv(filepath_or_buffer = PATH).head(5)
print(df_train.head(5))



def main_dpo_training():
    """
    Trains a DPO model with a fixed configuration for a single run without
    data partitioning loops or results file logging.
    """
    # --- Configuration ---
    PATH = 'gh114.csv'  # <-- IMPORTANT: Set your data path here
    
    # Fixed Hyperparameters
    N_EPOCHS = 1
    BETA = 0.01
    LEARNING_RATE = 1e-7
    EPSILON = 0.3  # Epsilon for constructing preference pairs

    # Model and Training Configuration
    OUTPUT_NAME = 'w_dpo_0'
    LOGGING_STEPS = 2
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    ADAM_DECAY = 0.1

    # --- Data Loading and Preparation ---
    try:
        df = pd.read_csv(filepath_or_buffer=PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{PATH}'. Creating a dummy file.")
        data = {
            'sequence': [f'SEQ_{k}' for k in range(30)],
            'target_reg': np.random.rand(30),
            'part_0': ([1]*10) + ([0]*20),
            'part_1': ([0]*10) + ([1]*10) + ([0]*10),
            'part_2': ([0]*20) + ([1]*10)
        }
        df = pd.DataFrame(data)
        df.to_csv(PATH, index=False)
        print(f"A dummy file '{PATH}' has been created. Please populate it and run again.")
        return

    # Manually define training and validation partitions
    df_part0 = df[(df['part_0'] == 1) & (df['part_1'] == 0) & (df['part_2'] == 0)]
    df_part1 = df[(df['part_1'] == 1) & (df['part_0'] == 0) & (df['part_2'] == 0)]
    df_part2 = df[(df['part_2'] == 1) & (df['part_0'] == 0) & (df['part_1'] == 0)]
    
    df_train = df_part0
    df_valid_merged = pd.concat([df_part1, df_part2], ignore_index=True)

    # --- Model and Tokenizer Initialization ---
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
    tokenizer = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")

    # --- Helper Function for Dataset Creation ---
    def prepare_dataset(df_prep, epsilon_prep):
        """Sorts dataframe, constructs pairs, and creates a Hugging Face Dataset."""
        df_sorted = df_prep.sort_values(by='target_reg', ascending=False)
        y_vec = df_sorted['target_reg'].to_list()
        seqs = df_sorted['sequence'].to_list()
        
        index_pairs, pair_diffs = construct_pairs_e(Yvec=y_vec, epsilon=epsilon_prep)
        
        data_dict = format_string_pairs_e(
            strings=seqs, 
            index_pairs=index_pairs, 
            pair_diffs=pair_diffs,
            N_characters=None
        )
        
        hf_dataset = Dataset.from_dict(data_dict)
        return hf_dataset, len(index_pairs)

    # --- Dataset Preparation ---
    print("Preparing consolidated validation dataset...")
    hf_validation_dataset, n_pairs_val = prepare_dataset(df_valid_merged, EPSILON)
    print(f"Consolidated validation dataset created with {n_pairs_val} pairs.")

    print("Preparing training dataset...")
    hf_train_dataset, n_pairs_train = prepare_dataset(df_train, EPSILON)
    print(f"Training dataset created with {n_pairs_train} pairs.")
    
    if n_pairs_train == 0:
        print("Stopping script: no training pairs were generated.")
        return

    # --- Configure and Train the DPO Model ---
    config_dict = {
        'output_dir': f"{OUTPUT_NAME}",
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
        'loss_type': 'w_sigmoid',
        'eval_strategy': "steps",
        'eval_steps': 2
    }
    
    training_args = DPOConfig(**config_dict)
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_validation_dataset,
        processing_class=tokenizer
    )
    
    print(f"\n--- Starting Single DPO Training Run ---")
    trainer.train()
    print("--- Training complete. ---")

    # # --- Evaluation (Placeholder) ---
    # print("Starting evaluation...")
    # eval_score = -1.0 # Placeholder, as custom logic is needed for DPO eval
    # print(f"Evaluation Score (placeholder): {eval_score:.4f}")
    
    # # --- Clean Up Memory ---
    # print("Cleaning up memory...")
    # del trainer, hf_train_dataset, hf_validation_dataset, model
    # gc.collect()
    # torch.cuda.empty_cache()

    # print('\n--- Script finished. ---')


if __name__ == '__main__':
    main_dpo_training()

raise NotImplementedError



    
# INPUT VARIABLES:
FILENAME = 'gh114.csv' # CSV for traniing and testing models
PATH = os.path.join(os.getcwd(), 'data','raw', FILENAME) # PATH to file
RESULT_NAME= 'train_df_result_test7.csv'

def get_logps(trainer : DPOTrainer):
    
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']

    return ref_chosen_logps, ref_rejected_logps

def calculate_accuracy(df):
    logps_condition = df['chosen_logps'] > df['rejected_logps']
    # Second condition: chosen_target_reg > rejected_target_reg
    target_condition = df['chosen_target_reg'] > df['rejected_target_reg']
    # Calculate accuracy
    accuracy = (logps_condition & target_condition).mean()

    return accuracy

def memory_stats():
    print("memory allocated: ", torch.cuda.memory_allocated()/1024**2)
    print("memory reserved: ", torch.cuda.memory_reserved()/1024**2)




if __name__ == "__main__":
    main()

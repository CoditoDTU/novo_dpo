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

    


def main():
    #1 Separate DFs
    df = pd.read_csv(filepath_or_buffer = PATH) 
    df_part0 = df[(df['part_0'] == 1) & (df['part_1'] == 0) & (df['part_2'] == 0)]#.head(4)
    df_part1 = df[(df['part_1'] == 1) & (df['part_0'] == 0) & (df['part_2'] == 0)]#.head(4)
    df_part2 = df[(df['part_2'] == 1) & (df['part_0'] == 0) & (df['part_1'] == 0)]#.head(2)
    dfs = [df_part0, df_part1, df_part2]

    # Load model and tokenizer once at the beginning
    MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
    TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")

    # Initialize results file if it doesn't exist
    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'n_epochs', 'betas', 'epsilon', 'learning_rate', 'P_train',
                'Eval_score_1', 'Eval_score_2', 'N_pairs', 'N_pairs_val_1', 'N_pairs_val_'
            ])

    # Load existing results to check for duplicates
    processed_combinations = set()
    if os.path.exists(RESULT_NAME):
        try:
            results_df = pd.read_csv(RESULT_NAME)
            # Create unique identifier combining id and P_train
            processed_combinations = set(zip(results_df['id'], results_df['P_train']))
        except:
            processed_combinations = set()

    for i, df_train in enumerate(dfs):
        if i == 10:
            print('process stopped')
            break 

        val_indices = [j for j in range(3) if j != i]
        df_valid_1 = dfs[val_indices[0]]
        df_valid_2 = dfs[val_indices[1]]
        df_val_merged = pd.concat([df_valid_1, df_valid_2], ignore_index=True).copy()
        df_val_merged_filt = df_val_merged[['sequence', 'target_reg']]
        
        # Load hyperparameters
        hyp_file = pd.read_csv(filepath_or_buffer='configs/hyperparameter_combinations_small_id.csv')

        OUTPUT_NAME = 'grid_test'
        LOGGING_STEPS = 1
        ADAM_BETAS = (0.9, 0.999)
        ADAM_EPSILON = 1e-8
        ADAM_DECAY = 0.1

        for _, row in hyp_file.iterrows():
            current_part = f'part_{i}'
            # Check if this specific combination (id + part) has already been processed
            if (row['id'], current_part) in processed_combinations:
                print(f"Skipping already processed combination - ID: {row['id']}, Part: {current_part}")
                continue

            epsilon = row['epsilons']

            # TRAIN (rest of your training code remains the same)
            df_sorted_train = df_train.sort_values(by='target_reg', ascending=False)
            y_train = df_sorted_train['target_reg'].to_list()
            seq_train = df_sorted_train['sequence'].to_list()
            pairs_train = construct_pairs(Yvec=y_train, epsilon=epsilon)
            train_dict = format_string_pairs(strings=seq_train, index_pairs=pairs_train, N_characters=None)
            hf_train_dataset = Dataset.from_dict(train_dict)
            N_pairs = len(pairs_train)

            
            # Helper function to generate validation data
            def prepare_validation(df_valid, epsilon_val):

                df_sorted_valid = df_valid.sort_values(by='target_reg', ascending=False)
                y_valid = df_sorted_valid['target_reg'].to_list()
                seq_valid = df_sorted_valid['sequence'].to_list()
                pairs_valid = construct_pairs(Yvec=y_valid, epsilon=epsilon_val)
                N_pairs_val = len(pairs_valid)
                valid_dict = format_string_pairs(strings=seq_valid, index_pairs=pairs_valid, N_characters=None)
                hf_dataset = Dataset.from_dict(valid_dict)

                return valid_dict, hf_dataset, N_pairs_val

            # VALIDATION 1
            valid_dict_1_e, hf_val1_dataset_e, N_pairs_val_1 = prepare_validation(df_valid_1, epsilon)
            #valid_dict_1_0, hf_val1_dataset_0 = prepare_validation(df_valid_1, 0)

            # VALIDATION 2
            valid_dict_2_e, hf_val2_dataset_e, N_pairs_val_2 = prepare_validation(df_valid_2, epsilon)
            #valid_dict_2_0, hf_val2_dataset_0 = prepare_validation(df_valid_2, 0)


            # Config dict:
            config_dict = {
                'output_dir': OUTPUT_NAME,
                'logging_steps': LOGGING_STEPS,
                'beta': row['betas'],
                'learning_rate': row['learning_rate'],
                'adam_beta1': ADAM_BETAS[0],
                'adam_beta2': ADAM_BETAS[1],
                'num_train_epochs': row['epochs'],
                'adam_epsilon': ADAM_EPSILON,
                'weight_decay': ADAM_DECAY,
                'precompute_ref_log_probs' : True,
                'report_to': 'none',
                'auto_find_batch_size' : True
            }
            
            # Train model
            training_args = DPOConfig(**config_dict)
            trainer = DPOTrainer(model=MODEL,
                                args=training_args,
                                train_dataset=hf_train_dataset,
                                processing_class=TOKENIZER)
            
            print(f'starting training in part {current_part} row with id {row['id']}')
            print(f'length of dictionary: {len(train_dict)}')
            trainer.train()
            # --- GET LOGPS FOR ALL 4 VALIDATION CONFIGS ---
            # Validation logps
            def compute_logps(trainer, valid_dict, hf_dataset):
                val_trainer = DPOTrainer(model=trainer.model,
                                        args=trainer.args,
                                        train_dataset=hf_dataset,
                                        processing_class=trainer.processing_class)
                
                chosen_logps, rejected_logps = get_logps(val_trainer)

                chosen_df = pd.DataFrame({
                    'chosen_sequences': valid_dict['chosen'],
                    'chosen_logps': chosen_logps
                })
                chosen_df = pd.merge(
                    chosen_df, df_val_merged_filt,
                    left_on='chosen_sequences', right_on='sequence', how='left'
                ).rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])

                rejected_df = pd.DataFrame({
                    'rejected_sequences': valid_dict['rejected'],
                    'rejected_logps': rejected_logps
                })
                rejected_df = pd.merge(
                    rejected_df, df_val_merged_filt,
                    left_on='rejected_sequences', right_on='sequence', how='left'
                ).rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])

                summary_df = pd.concat([chosen_df, rejected_df], axis=1)
                return calculate_accuracy(summary_df)
            
            print('starting validation')
            # Accuracy for all validations
            print('starting validation 1')
            accuracy_val_1_epsilon = compute_logps(trainer, valid_dict_1_e, hf_val1_dataset_e)
            accuracy_val_2_epsilon = compute_logps(trainer, valid_dict_2_e, hf_val2_dataset_e)

            # --- WRITE RESULTS TO CSV ---
            with open(RESULT_NAME, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    row['id'],
                    row['epochs'],
                    row['betas'],
                    epsilon,
                    row['learning_rate'],
                    current_part,  # This is f'part_{i}'
                    float(accuracy_val_1_epsilon),
                    float(accuracy_val_2_epsilon),
                    N_pairs,
                    N_pairs_val_1,
                    N_pairs_val_2
                ])
            
            # Add this combination to the processed set
            processed_combinations.add((row['id'], current_part))
            
            #print('\nMemory stats before release:',memory_stats())

            del trainer
            #  del model
            gc.collect()
            torch.cuda.empty_cache()

            #print('\nMemory stats after release:',memory_stats())
    print('DONE UwU')





if __name__ == "__main__":
    main()

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

    
# INPUT VARIABLES:
FILENAME = 'gh114.csv' # CSV for traniing and testing models
PATH = os.path.join(os.getcwd(), 'data','raw', FILENAME) # PATH to file


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



    


def main():
        #1 Separate DFs
    df = pd.read_csv(filepath_or_buffer = FILENAME) 
    df_part0 = df[(df['part_0'] == 1) & (df['part_1'] == 0) & (df['part_2'] == 0)]
    df_part1 = df[(df['part_1'] == 1) & (df['part_0'] == 0) & (df['part_2'] == 0)]
    df_part2 = df[(df['part_2'] == 1) & (df['part_0'] == 0) & (df['part_1'] == 0)]
    dfs = [df_part0, df_part1, df_part2]

    results = []

    for i, df_train in enumerate(tqdm(dfs, desc="Processing data partitions")):
        val_indices = [j for j in range(3) if j != i]
        df_valid_1 = dfs[val_indices[0]]
        df_valid_2 = dfs[val_indices[1]]
        df_val_merged = pd.concat([df_valid_1, df_valid_2], ignore_index=True).copy() # merged DF to get metric obs
        df_val_merged_filt = df_val_merged[['sequence',  # Keep only interested columns
                                            'target_reg']]
        # Load hyperparameters and model
        hyp_file = pd.read_csv(filepath_or_buffer='configs/hyperparameter_combinations_small.csv')
        OUTPUT_NAME = 'grid_test'
        LOGGING_STEPS = 1
        ADAM_BETAS = (0.9, 0.999)
        ADAM_EPSILON = 1e-8
        ADAM_DECAY = 0.1
        MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
        TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
        

        for _, row in tqdm(hyp_file.iterrows(), total=len(hyp_file), desc=f"Partition {i} hyperparameters", leave=False):
            #print(row)
            epsilon = row['epsilons']

            # TRAIN
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
                valid_dict = format_string_pairs(strings=seq_valid, index_pairs=pairs_valid, N_characters=None)
                hf_dataset = Dataset.from_dict(valid_dict)

                return valid_dict, hf_dataset

            # VALIDATION 1
            valid_dict_1_e, hf_val1_dataset_e = prepare_validation(df_valid_1, epsilon)
            valid_dict_1_0, hf_val1_dataset_0 = prepare_validation(df_valid_1, 0)

            # VALIDATION 2
            valid_dict_2_e, hf_val2_dataset_e = prepare_validation(df_valid_2, epsilon)
            valid_dict_2_0, hf_val2_dataset_0 = prepare_validation(df_valid_2, 0)


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
                'report_to': 'none'
            }
            
            # Train model
            training_args = DPOConfig(**config_dict)
            trainer = DPOTrainer(model=MODEL,
                                args=training_args,
                                train_dataset=hf_train_dataset,
                                processing_class=TOKENIZER)
            trainer.train()
            
            # --- GET LOGPS FOR ALL 4 VALIDATION CONFIGS ---
            # Validation logps
            def compute_logps(valid_dict, hf_dataset):
                val_trainer = DPOTrainer(model=MODEL,
                                        args=DPOConfig(**config_dict),
                                        train_dataset=hf_dataset,
                                        processing_class=TOKENIZER)
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
                del val_trainer
                torch.cuda.empty_cache()
                return calculate_accuracy(summary_df)
            

            # Accuracy for all validations
            accuracy_val_1_epsilon = compute_logps(valid_dict_1_e, hf_val1_dataset_e)
            accuracy_val_1_epsilon0 = compute_logps(valid_dict_1_0, hf_val1_dataset_0)

            accuracy_val_2_epsilon = compute_logps(valid_dict_2_e, hf_val2_dataset_e)
            accuracy_val_2_epsilon0 = compute_logps(valid_dict_2_0, hf_val2_dataset_0)

            # --- APPEND RESULTS TO RESULTS ---
            # Append to results
            results.append({
                'n_epochs': row['epochs'],
                'betas': row['betas'],
                'epsilon': epsilon,
                'learning_rate': row['learning_rate'],
                'P_train': f'part_{i}',
                'Eval_score_1': ((val_indices[0], accuracy_val_1_epsilon0), (val_indices[0], accuracy_val_1_epsilon)),
                'Eval_score_2': ((val_indices[1], accuracy_val_2_epsilon0), (val_indices[1], accuracy_val_2_epsilon)),
                'N_pairs': N_pairs
            })
            del trainer
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
            
    results_df.to_csv('train_df_result_small.csv')






if __name__ == "__main__":
    main()

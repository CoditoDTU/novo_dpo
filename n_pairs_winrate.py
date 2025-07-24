import pandas as pd
import numpy as np
import os
import sys
import random
from typing import Dict, List, Tuple
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm.auto import tqdm
import csv
import torch
import gc
import argparse
import math
import shutil
 
# --- Helper Functions ---
 
def calculate_winrate(df: pd.DataFrame) -> float:
    """Calculates the winrate (accuracy)."""
    model_prefers_chosen = pd.to_numeric(df['chosen_logps']) > pd.to_numeric(df['rejected_logps'])
    target_is_ordered = pd.to_numeric(df['chosen_target_reg']) > pd.to_numeric(df['rejected_target_reg'])
    winrate = model_prefers_chosen[target_is_ordered].mean()
    return winrate
 
def get_logps(trainer: DPOTrainer) -> Tuple[List[float], List[float]]:
    """Gets the pre-computed reference log probabilities from a trainer instance."""
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']
    return ref_chosen_logps, ref_rejected_logps
 
def get_initial_logps_with_trainer(sequences: List[str], model, tokenizer) -> Dict[str, float]:
    """
    Calculates initial logps using the DPOTrainer mechanism, ensuring full cleanup.
    """
    print("  Calculating initial log probabilities for training set...")
 
    unique_sequences = sorted(list(set(sequences)))
    num_seqs = len(unique_sequences)
    dummy_dict = {"prompt": ["M"] * num_seqs, "chosen": unique_sequences, "rejected": [""] * num_seqs}
    dummy_dataset = Dataset.from_dict(dummy_dict)
 
    temp_dir = os.path.join(os.getcwd(), "temp_dpo_logps_inference")
 
    temp_config = DPOConfig(output_dir=temp_dir, precompute_ref_log_probs=True, report_to='none', save_safetensors=False)
 
    temp_trainer = DPOTrainer(model=model, args=temp_config, train_dataset=dummy_dataset, processing_class=tokenizer)
 
    chosen_logps, _ = get_logps(temp_trainer)
 
    logp_map = {seq: logp for seq, logp in zip(unique_sequences, chosen_logps)}
 
    del chosen_logps
    del temp_trainer
    del temp_config
    del dummy_dataset
 
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
 
    return logp_map
 
 
def construct_pairs_e(Yvec, epsilon):
    '''Constructs and sorts pairs of indexes by score difference. Assumes Yvec is sorted.'''
    Yvec = np.asarray(Yvec)
    i_idx, j_idx = np.triu_indices(len(Yvec), k=1)
    diff = Yvec[i_idx] - Yvec[j_idx]
    mask = diff > epsilon
    valid_pairs_indices = np.stack([i_idx[mask], j_idx[mask]], axis=1).astype(int)
    valid_diffs = diff[mask]
    combined = sorted(zip(valid_pairs_indices, valid_diffs), key=lambda x: x[1], reverse=True)
    return combined
 
def format_string_pairs_e(strings: List[str], all_pairs_data: List[Tuple[Tuple[int, int], float]], N_characters: int = None) -> Dict[str, List]:
    '''Formats a dictionary for DPO.'''
    result = {"prompt": [], "chosen": [], "rejected": [], "epsilon": []}
    force_prompt = 'M' if N_characters is None else None
 
    for (i, j), diff_value in all_pairs_data:
        chosen_str, rejected_str = strings[i], strings[j]
        prompt = force_prompt if force_prompt else chosen_str[:N_characters]
        chosen = chosen_str if force_prompt else chosen_str[N_characters:]
        rejected = rejected_str if force_prompt else rejected_str[N_characters:]
 
        result["prompt"].append(prompt)
        result["chosen"].append(chosen)
        result["rejected"].append(rejected)
        result["epsilon"].append(diff_value)
    return result
 
def calculate_n_prime(n: int) -> int:
    """Calculates n' from n using the formula n(n-1)/2."""
    if n < 2:
        return 0
    return math.comb(n, 2)
 
def print_dry_run_plan(modalities: List[str], n_values: List[int], replicates: int, loss_type: str, result_name: str):
    """Includes all five methods in the plan."""
    print("\n--- 🧪 Experiment Dry Run Plan 🧪 ---")
    print(f"  - Methods to run: {modalities}")
    print(f"  - DPO Loss Type: {loss_type}")
    print(f"  - Replicates per run: {replicates}")
    print(f"  - Results will be saved to: '{result_name}'")
    print(f"  - Model checkpoints will NOT be saved.")
 
    print("\n--- Training Schedule ---")
    total_models = len(modalities) * len(n_values) * replicates
 
    for modality in modalities:
        print(f"\n  For Method: '{modality}'")
        for n in n_values:
            if modality == 'IID_partition':
                print(f"    - n = {n:2d} rows sampled -> {replicates} models will be trained.")
            else:
                n_prime = calculate_n_prime(n)
                print(f"    - n = {n:2d} (n' = {n_prime} pairs) -> {replicates} models will be trained.")
 
    print("-----------------------------------")
    print(f"  - Total models to be trained: {total_models}")
 
    print("\n--- Dry run complete. Exiting. ---\n")
 
 
# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run DPO fine-tuning with multiple sampling strategies.")
    parser.add_argument('--input_file', type=str, default='gh114.csv', help='Path to the input CSV file.')
    parser.add_argument('--n_run', type=int, required=True, help='The experiment run number for directory naming.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Minimum score difference for training pairs.')
    parser.add_argument('--replicates', type=int, default=10, help='Number of replicates for each value of n.')
    parser.add_argument('--dry-run', action='store_true', help='Prints the experiment plan and exits without running.')
    args = parser.parse_args()
 
    # --- Configuration ---
    N_VALUES = [2, 4, 8, 16, 32]
    LOSS_TYPE = 'sigmoid'
    BETA = 0.01
    LEARNING_RATE = 1e-5
    N_EPOCHS = 1
    MODALITIES = ['IID_partition', 'max_discrepancy', 'min_discrepancy', 'max_logps', 'min_logps']
    LOGGING_STEPS = 2
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    ADAM_DECAY = 0.1
 
    OUTPUT_NAME = os.path.splitext(args.input_file)[0]
    PARENT_OUTPUT_DIR = os.path.join(os.getcwd(), 'DPO_exp_all_methods', OUTPUT_NAME,LOSS_TYPE, f'run_{args.n_run}')
    RESULT_NAME = os.path.join(PARENT_OUTPUT_DIR, f'winrate_exp_{LOSS_TYPE}_{args.n_run}.csv')
    os.makedirs(PARENT_OUTPUT_DIR, exist_ok=True)
 
    if args.dry_run:
        print_dry_run_plan(MODALITIES, N_VALUES, args.replicates, LOSS_TYPE, RESULT_NAME)
        sys.exit(0)
 
    # --- Data Preparation ---
    print("--- Preparing data ---")
    df_full = pd.read_csv(filepath_or_buffer=args.input_file)
    df_full['partition'] = df_full[['part_0', 'part_1', 'part_2']].idxmax(axis=1).str.extract(r'(\d+)').astype(int)
    partitions = sorted(df_full['partition'].unique())
 
    if not os.path.exists(RESULT_NAME):
        with open(RESULT_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['modality', 'replicate', 'n_samples', 'n_train_pairs', 'n_validation_pairs', 'winrate', 'loss_type'])
 
    TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
    if TOKENIZER.pad_token is None: TOKENIZER.pad_token = TOKENIZER.eos_token
 
    # --- Experiment Loop ---
    for n_samples in N_VALUES:
        for replicate in range(1, args.replicates + 1):
            print(f"\n{'='*40}\nProcessing n={n_samples}, Replicate={replicate}/{args.replicates}\n{'='*40}")
 
            # ✨ MODIFIED: Partitions are now fixed, not randomly selected.
            # train_partition_ids = [2, 1]
            # val_partition_id = 0
            val_partition_ids = [0,1]
            train_partition_id = 2
 
            #df_train_full = df_full[df_full['partition'].isin(train_partition_ids)].copy()
            df_train_full = df_full[df_full['partition'] == train_partition_id].copy()
 
            # df_val = df_full[df_full['partition'] == val_partition_id]
            df_val = df_full[df_full['partition'].isin(val_partition_ids)].copy()
 
            df_val_sorted = df_val.sort_values(by='target_reg', ascending=False).reset_index(drop=True)
            hf_val_dataset = Dataset.from_dict(format_string_pairs_e(
                df_val_sorted['sequence'].to_list(),
                construct_pairs_e(df_val_sorted['target_reg'].to_list(), 0.0) # Validation epsilon
            ))
            n_valid_pairs = len(hf_val_dataset)
        # ✨ MODIFIED: Calculate reference logps for ALL sequences (train + val) at once
            print("--- Pre-calculating all reference log probabilities ---")
            base_model = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
            all_unique_sequences = list(set(df_train_full['sequence'].to_list() + df_val['sequence'].to_list()))
            logp_map = get_initial_logps_with_trainer(all_unique_sequences, base_model, TOKENIZER)
            
            df_train_full['logps'] = df_train_full['sequence'].map(logp_map)
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
 
            for modality in MODALITIES:
                print(f"\n--- Training: Modality={modality}, n={n_samples}, Replicate={replicate}/{args.replicates} ---")
 
                train_pairs = []
                train_sequences = []
 
                if modality == 'IID_partition':
                    df_train_sample = df_train_full.sample(n=n_samples, random_state=replicate)
                    df_train_sample_sorted = df_train_sample.sort_values(by='target_reg', ascending=False).reset_index(drop=True)
                    train_sequences = df_train_sample_sorted['sequence'].to_list()
                    train_scores = df_train_sample_sorted['target_reg'].to_list()
                    train_pairs = construct_pairs_e(Yvec=train_scores, epsilon=args.epsilon)
 
                elif modality in ['max_discrepancy', 'min_discrepancy']:
                    n_prime = calculate_n_prime(n_samples)
                    df_train_sorted = df_train_full.sort_values(by='target_reg', ascending=False).reset_index(drop=True)
                    train_sequences = df_train_sorted['sequence'].to_list()
                    all_pairs = construct_pairs_e(df_train_sorted['target_reg'].to_list(), args.epsilon)
 
                    if len(all_pairs) < n_prime: continue
                    train_pairs = all_pairs[:n_prime] if modality == 'max_discrepancy' else all_pairs[-n_prime:]
 
                elif modality in ['max_logps', 'min_logps']:
                    n_prime = calculate_n_prime(n_samples)
                    df_train_logp_sorted = df_train_full.sort_values(by='logps', ascending=False).reset_index(drop=True)
                    train_sequences = df_train_logp_sorted['sequence'].to_list()
 
                    logp_pairs = construct_pairs_e(df_train_logp_sorted['logps'].to_list(), args.epsilon)
                    if len(logp_pairs) < n_prime: continue
                    selected_pairs_by_logp = logp_pairs[:n_prime] if modality == 'max_logps' else logp_pairs[-n_prime:]
 
                    target_reg_scores = df_train_logp_sorted['target_reg'].to_list()
                    train_pairs = [((i, j), target_reg_scores[i] - target_reg_scores[j]) for (i, j), _ in selected_pairs_by_logp]
 
                if not train_pairs:
                    print(f"--> Skipping: No training pairs generated for this configuration.")
                    continue
 
                hf_train_dataset = Dataset.from_dict(format_string_pairs_e(train_sequences, train_pairs))
                n_train_pairs = len(hf_train_dataset)
 
                model = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
                model_output_dir = os.path.join(PARENT_OUTPUT_DIR,'models', f"model_{modality}_n{n_samples}_rep{replicate}")
 
                config_dict = {
                'output_dir': OUTPUT_NAME,
                'logging_steps': LOGGING_STEPS,
                'beta': BETA,
                'learning_rate': LEARNING_RATE,
                'adam_beta1': ADAM_BETAS[0],
                'adam_beta2': ADAM_BETAS[1],
                'num_train_epochs': N_EPOCHS,
                'adam_epsilon': ADAM_EPSILON,
                'weight_decay': ADAM_DECAY,
                'precompute_ref_log_probs' : True,
                'report_to': 'none',
                'auto_find_batch_size' : True,
                'save_strategy': 'no',
                'save_safetensors': False,
                'remove_unused_columns': False,
                'report_to':None
            }  
                training_args = DPOConfig(**config_dict)
                # training_args = DPOConfig(
                #     output_dir=model_output_dir, beta=BETA, learning_rate=LEARNING_RATE,
                #     num_train_epochs=N_EPOCHS, loss_type=LOSS_TYPE, precompute_ref_log_probs=True,
                #     report_to='none', auto_find_batch_size=True, remove_unused_columns=False,
                #     save_strategy="no", save_safetensors=False
                # )
                trainer = DPOTrainer(model=model, args=training_args, train_dataset=hf_train_dataset, processing_class=TOKENIZER)
 
                print('###############################Training Starting##################################')
                trainer.train()
                print('###############################Training Finished###################################################')
 
 
                print('Starting validation...')
                val_trainer = DPOTrainer(model=trainer.model, args=trainer.args, train_dataset=hf_val_dataset, processing_class=trainer.processing_class)
                chosen_logps, rejected_logps = get_logps(val_trainer)

                # ✨ MODIFIED: Get reference logps from the pre-computed map
                ref_chosen_logps = [logp_map[seq] for seq in hf_val_dataset['chosen']]
                ref_rejected_logps = [logp_map[seq] for seq in hf_val_dataset['rejected']]

                # ✨ MODIFIED: Build the summary dataframe with all requested columns
                df_val_merged = df_val_sorted[['sequence', 'target_reg']].drop_duplicates()
                chosen_df = pd.DataFrame({
                    'chosen_sequences': hf_val_dataset['chosen'],
                    'ref_chosen_logps': ref_chosen_logps,
                    'chosen_logps': chosen_logps
                })
                chosen_df = pd.merge(chosen_df, df_val_merged, left_on='chosen_sequences', right_on='sequence').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])

                rejected_df = pd.DataFrame({
                    'rejected_sequences': hf_val_dataset['rejected'],
                    'rejected_logps': rejected_logps,
                    'ref_rejected_logps': ref_rejected_logps
                })

                rejected_df = pd.merge(rejected_df, df_val_merged, left_on='rejected_sequences', right_on='sequence').rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])
                

                summary_df_raw = pd.concat([chosen_df, rejected_df], axis=1).dropna()
                final_column_order = [
                    'chosen_sequences', 'ref_chosen_logps', 'chosen_logps', 'chosen_target_reg',
                    'rejected_sequences', 'rejected_logps', 'ref_rejected_logps', 'rejected_target_reg'
                ]
                summary_df = summary_df_raw[final_column_order]

 
                summary_filepath = os.path.join(model_output_dir, f'{modality}_replicate_{replicate}_n_{n_samples}summary_df.csv')
                summary_df.to_csv(summary_filepath, index=False)
                print(f"Validation summary saved to: {summary_filepath}")
 
                if not summary_df.empty:
                    winrate = calculate_winrate(summary_df)
                    print(f'--> Winrate: {winrate:.4f}')
                    with open(RESULT_NAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([modality, replicate, n_samples, n_train_pairs, n_valid_pairs, float(winrate), LOSS_TYPE])
 
                del trainer, model, val_trainer, summary_df, hf_train_dataset
                gc.collect()
                torch.cuda.empty_cache()
 
    print('\nDONE OwO/')
 
if __name__ == "__main__":
    main()
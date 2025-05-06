'''
Script to evaluate a trained HF model with a validation dataset

Add option to decide if the DFs will have a prompt or if it was added an M for the prompt
if this was the case, then just grab all the sequence column instead of using a prompt
'''

# %%
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import sys
import json
from datasets import Dataset
from datasets import DatasetDict
import torch
import matplotlib.pyplot as plt
import pandas as pd
#from src.pyutils.data_utils import *
# DPO Trainer args

OG_DF = os.path.join(os.getcwd(), 'data/raw/gh114.csv')



VAL_FILENAMES = ['gh114_0_train.json',
                 'gh114_1_train.json',
                 'gh114_2_train.json']

VAL_DATA_PATHS = [os.path.join(os.getcwd(), "src", "pyutils", VAL_FILENAMES[0]),
                 os.path.join(os.getcwd(), "src", "pyutils", VAL_FILENAMES[1]),
                 os.path.join(os.getcwd(), "src", "pyutils", VAL_FILENAMES[2])]

PLOT_NAME = 'logps_DMS_2.pdf'





#TEST_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TEST_FILENAME)
PLOT_PATH = os.path.join("data", "processed", PLOT_NAME)

MODEL_NAMES =["NorseDrunkenSailor/ProtGPT2-with-pad_gh114_train_p0",
               "NorseDrunkenSailor/ProtGPT2-with-pad_gh114_train_p1",
               "NorseDrunkenSailor/ProtGPT2-with-pad_gh114_train_p2"]

# MODELS = [AutoModelForCausalLM.from_pretrained(MODEL_NAMES[0]),
#           AutoModelForCausalLM.from_pretrained(MODEL_NAMES[1]),
#           AutoModelForCausalLM.from_pretrained(MODEL_NAMES[2])]
 
TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad_oxda0")

# DPO config args

OUTPUT_NAME = 'DPO_protgpt2_oxda_logprobstest'
EVAL_JSON_PATH = os.path.join(os.getcwd(), OUTPUT_NAME,'evaluate_dict_dpo_4.json' )
LOGGING_STEPS = 1
BETA = 0.1
LEARNING_RATE = 1e-5
ADAM_BETAS = (0.9, 0.999) 
ADAM_EPSILON = 1e-8
N_TRAIN_EPOCHS = 1
ADAM_DECAY = 0.1

# %%
# Config dict
config_dict = {
    'output_dir': OUTPUT_NAME,
    'logging_steps': LOGGING_STEPS,
    'beta': BETA,
    'learning_rate': LEARNING_RATE,
    'adam_beta1': ADAM_BETAS[0],
    'adam_beta2': ADAM_BETAS[1],
    'num_train_epochs': N_TRAIN_EPOCHS,
    'adam_epsilon': ADAM_EPSILON,
    'weight_decay': ADAM_DECAY,
    'precompute_ref_log_probs' : True
}


def get_logps(trainer : DPOTrainer):
    
    trainer.get_train_dataloader()
    ref_chosen_logps = trainer.train_dataset['ref_chosen_logps']
    ref_rejected_logps = trainer.train_dataset['ref_rejected_logps']

    return ref_chosen_logps, ref_rejected_logps


def plot_logps(chosen_logps: list[int], rejected_logps: list[int], path: str):
    """
    Plots chosen_logps and rejected_logps against each other and saves the plot to the specified path.

    Parameters:
    - chosen_logps (list of int): List of chosen log probabilities.
    - rejected_logps (list of int): List of rejected log probabilities.
    - path (str): File path to save the plot image.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rejected_logps, chosen_logps, label='Chosen Logps', marker='o')

    # Compute min/max range across all points
    all_values = chosen_logps + rejected_logps
    min_val = min(all_values)
    max_val = max(all_values)
    value_range = max_val - min_val
    margin = value_range * 0.1

    # Define the y = x line range
    line_min = min_val - margin
    line_max = max_val + margin
    plt.plot([line_min, line_max], [line_min, line_max], 'r--', label='y = x')

    # Labeling and aesthetics
    plt.xlabel('Rejected Log Probability')
    plt.ylabel('Chosen Log Probability')
    plt.title('Chosen vs Rejected Log Probabilities')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_logps_vs_dms(df, save_path):
    """
    Plots a scatter plot of chosen_logps vs DMS_score from a given DataFrame,
    and saves the plot to the specified path.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'chosen_logps' and 'DMS_score'.
        save_path (str): File path to save the plot (e.g. 'plot.png').
    """
    chosen_logps = df['chosen_logps']
    dms_scores = df['DMS_score']
    plt.figure(figsize=(8, 6))
    plt.scatter(chosen_logps, dms_scores, alpha=0.7)

    # # Compute min/max range across all points
    # all_values = chosen_logps + dms_scores
    # min_val = min(all_values)
    # max_val = max(all_values)
    # value_range = max_val - min_val
    # margin = value_range * 0.1

    # # Define the y = x line range
    # line_min = min_val - margin
    # line_max = max_val + margin
    # plt.plot([line_min, line_max], [line_min, line_max], 'r--', label='y = x')


    plt.xlabel('Chosen LogPs')
    plt.ylabel('DMS Score')
    plt.title('DMS Score vs Chosen LogPs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # closes the figure to free up memory

# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path = "NorseDrunkenSailor/ProtGPT2-with-pad_gh114_train_p0")



def main():
   

    #flop_df = pd.read_csv(filepath_or_buffer = OG_DF )
    for i in range(len(VAL_DATA_PATHS)):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAMES[i])

        # Load DF with desired metric
        flop_df = pd.read_csv(filepath_or_buffer = OG_DF)
        col_name = f"part_{i}"
        filtered_flop_df = flop_df[flop_df[col_name] == 0].copy() # we keep the datasets that have not being trained on 
        
        subset_flop_df = filtered_flop_df[['sequence',
                                           'target_reg']]

        #print(filtered_flop_df.head(n=10))
        for j in range(len(VAL_DATA_PATHS)):
            if j == i:
                continue
            # Load Validation dataset
            with open(VAL_DATA_PATHS[j], "r") as f:
                validation_dataset = json.load(f)
            
            chosen_sequences = validation_dataset['chosen']
            rejected_sequences = validation_dataset['rejected']

            hf_val_dataset = Dataset.from_dict(validation_dataset)

            #Create trainer for computing logits
            training_args = DPOConfig(**config_dict)
            trainer = DPOTrainer(model = model,
                         args = training_args,
                         train_dataset = hf_val_dataset,
                         processing_class = TOKENIZER)
            
            # create logit dfs
            chosen_logps, rejected_logps = get_logps(trainer = trainer)

            chosen_df = pd.DataFrame({
                'chosen_sequences' : chosen_sequences,
                'chosen_logps': chosen_logps
            })

            # Merge with subset_flop_df to get target_reg for chosen sequences
            chosen_df = pd.merge(
                chosen_df,
                subset_flop_df,
                left_on='chosen_sequences',
                right_on='sequence',
                how='left').rename(columns={'target_reg': 'chosen_target_reg'}).drop(columns=['sequence'])


            # Create rejected_df with target_reg
            rejected_df = pd.DataFrame({
                'rejected_sequences' : rejected_sequences,
                'rejected_logps': rejected_logps
            })

            # Merge with subset_flop_df to get target_reg for rejected sequences
            rejected_df = pd.merge(
                rejected_df,
                subset_flop_df,
                left_on='rejected_sequences',
                right_on='sequence',
                how='left'
            ).rename(columns={'target_reg': 'rejected_target_reg'}).drop(columns=['sequence'])

            # Combine into summary_df
            summary_df = pd.concat([chosen_df, rejected_df], axis=1)

            summary_df.to_csv(f'dpo_eval_df_p{i}_json{j}.csv')

            

            # plot_logps(chosen_logps = chosen_logps,
            #            rejected_logps = rejected_logps,
            #            path =f"data/processed/dpo_modelp{i}_val{j}.pdf")

        torch.cuda.empty_cache()

            

            # chosen_full_df = pd.merge(
            #     chosen_df,
            #     subset_flop_df,
            #     left_on='chosen_sequences',
            #     right_on='sequence',
            #     how='inner'
            #     )[['chosen_sequences', 'chosen_logps', 'target_reg']]
            
            # chosen_full_df.to_csv('chosen_full_df_p0_json1.csv')

       

        



    raise NotImplementedError
    prompt = "HSQKR"
    # Filter the DataFrame to sequences that start with 'HSQK'
    filtered_oxda = oxda_df[oxda_df['mutated_sequence'].str.startswith(prompt)].copy()
    # Create the new column with 'HSQK' removed
    filtered_oxda['completion_sequence'] = filtered_oxda['mutated_sequence'].str.removeprefix(prompt)

    subset_df = filtered_oxda[['mutated_sequence', 'completion_sequence', 'DMS_score']]


    with open(TRAIN_DATA_PATH, "r") as f:
        train_dataset = json.load(f)

    chosen_sequences = train_dataset['chosen']
    rejected_sequences = train_dataset['rejected']

    hf_train_dataset = Dataset.from_dict(train_dataset)

    training_args = DPOConfig(**config_dict)

    trainer = DPOTrainer(model = MODEL,
                         args = training_args,
                         train_dataset = hf_train_dataset,
                         processing_class = TOKENIZER)
    
    chosen_logps, rejected_logps = get_logps(trainer = trainer)

    chosen_df = pd.DataFrame({
        'chosen_sequences' : chosen_sequences,
        'chosen_logps': chosen_logps
    })

    rejected_df = pd.DataFrame({
        'rejected_sequences' : rejected_sequences,
        'rejected_logps': rejected_logps
    })


    chosen_full_df = pd.merge(
        chosen_df,
        subset_df,
        left_on='chosen_sequences',
        right_on='completion_sequence',
        how='inner'
        )[['mutated_sequence', 'chosen_sequences', 'chosen_logps', 'DMS_score']]
    
    print(len(chosen_df))
    print(len(subset_df))
    print(len(chosen_full_df))
    # print(chosen_df.head())
    # print(rejected_df.head())

    #print(Chosen_full_df.head())

    plot_logps_vs_dms(df = chosen_full_df,
                      save_path = PLOT_PATH)


    raise NotImplementedError


    oxda_df = pd.read_csv(filepath_or_buffer = 'data/raw/OXDA_RHOTO_Vanella_2023_activity.csv')
    prompt = "HSQK"
    # Filter the DataFrame to sequences that start with 'HSQK'
    filtered_oxda = oxda_df[oxda_df['mutated_sequence'].str.startswith(prompt)].copy()
    # Create the new column with 'HSQK' removed
    filtered_oxda['completion_sequence'] = filtered_oxda['mutated_sequence'].str[len('HSQK'):]

    subset_df = filtered_oxda[['mutated_sequence', 'completion_sequence', 'DMS_score']]
    

    print(f"Saving plot to: {PLOT_PATH}")
    plot_logps(chosen_logps = ref_chosen_logps,
               rejected_logps = ref_rejected_logps,
               path = PLOT_PATH)
    


    
    

main()
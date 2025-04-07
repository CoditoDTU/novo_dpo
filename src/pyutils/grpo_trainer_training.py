import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_tools import *
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from copy import deepcopy
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import os 
import sys
import json
from datasets import Dataset
from datasets import DatasetDict

#HYPER PARAMETERS
PATH = "/home/developer/Projects/novo_dpo/outputs/"
INPUT = "<|endoftext|>M"  # length is expressed in tokens, where each token has an average length of 4 amino acids.
N_SEQS = 10
MAX_LEN = 30
MODEL_NAME = "nferruz/ProtGPT2"
MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


# GRPO HYPERPARAMETERS
EPSILON = 0.01 # Clip error
BETA = 0.1 #KLD penalty
MU = 5  # Number of GRPO iterations
I = 2  # Total number of iterations
M = 3   # Number of steps per iteration
REFERENCE_VALUE = 7 # Propertie to optimize Ex: pI, hydrophibicity, pH, Temp
LERANING_RATE = 1e-6 # Learning rate for optimizer
#PLOTTING

REWARDS = []
LOSSES = []
GRADIENT_NORMS = []

# DATA PARAMETERS
FILENAME = "antimicrobial_peptides_dictionary2.json"

#DATA_PATH = os.path.join("home/developer/Projects/novo_dpo", 'data', 'processed', FILENAME)
TRAIN_DATA_PATH = "/home/developer/Projects/novo_dpo/data/processed/amp_train.json"#antimicrobial_peptides_dictionary2.json"
TEST_DATA_PATH = "/home/developer/Projects/novo_dpo/data/processed/amp_test.json"
# Optimizer setup (As mentioned in deepseek math paper)
optimizer = Adam(MODEL.parameters(),
                 lr = LERANING_RATE,
                 betas =(0.9, 0.95),
                 weight_decay = 0.1)

 
def main():
        
    # #dataset = load_dataset("trl-lib/tldr", split="train")

    # data_files = {"train": DATA_PATH, "test": DATA_PATH}

    # dataset = load_dataset("json", data_files = data_files, split="train")
    with open(TRAIN_DATA_PATH, "r") as f:
        train_dataset = json.load(f)

    with open(TEST_DATA_PATH, "r") as f:
        test_dataset = json.load(f)

    # Load into HuggingFace dataset
    hf_train_dataset = Dataset.from_dict(train_dataset)
    hf_test_dataset = Dataset.from_dict(test_dataset)


    training_args = GRPOConfig(output_dir="Protgpt2_test",
                           logging_steps=1,
                           report_to="none",
                           num_iterations = 10)

    trainer = GRPOTrainer(
        model="NorseDrunkenSailor/ProtGPT2-with-pad",
        reward_funcs=reward_hydrophobicity,
        args=training_args,
        train_dataset=hf_train_dataset,
        eval_dataset =hf_test_dataset,
    )

    trainer.train()

    raise NotImplementedError
    TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
    dataset = load_dataset("trl-lib/tldr", split="train")
    
    generated_sequences = pretrained_inf(model = MODEL,
                                         tokenizer = TOKENIZER,
                                        condition_tag = INPUT, 
                                        num_seqs = N_SEQS,
                                        max_length = MAX_LEN)
    
    seq_tuples = parsing_seqs_tuples(generated_sequences)

    trainer_input = [{'prompt': item[1]} for item in seq_tuples]

    training_args = GRPOConfig(output_dir = PATH, 
                               logging_steps = 10)
    
    trainer = GRPOTrainer(model = MODEL_NAME,
                          reward_funcs = compute_rewards,
                          args = training_args,
                          train_dataset = trainer_input)
    
    trainer.train()
if __name__ == "__main__":
    main()
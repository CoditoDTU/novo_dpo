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


#PATHS
TRAIN_FILENAME = 'amp_train.json'
TEST_FILENAME = 'amp_test.json'
TRAIN_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TRAIN_FILENAME)
TEST_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TEST_FILENAME)
OUTPUT_NAME = "protgpt2_test_rgd_3"

#HYPERPARAMETERS:
LEARNING_RATE = 1e-8
LOGGING_STEPS = 1
NUM_GENERATIONS = 20
NUM_ITERATIONS = 10 # mu in the GRPO algorithm
MAX_COMPLETION = 32 #


# ARGUMENTS
config_dict = {
    'output_dir': OUTPUT_NAME,
    'logging_steps': LOGGING_STEPS,
    'learning_rate': LEARNING_RATE,
    'num_generations': NUM_GENERATIONS,
    'num_iterations': NUM_ITERATIONS, 
    'remove_unused_columns': True,
    'log_completions': True,
    'max_completion_length' : MAX_COMPLETION,
    'per_device_train_batch_size': 100,   # by default 8. it is the M in the GRPO paper. Num of generation must evenly divide it. 
}


def main():
    #print(TRAIN_DATA_PATH)
    
    # LOAD DATASETS

    with open(TRAIN_DATA_PATH, "r") as f:
        train_dataset = json.load(f)

    with open(TEST_DATA_PATH, "r") as f:
        test_dataset = json.load(f)

    # COMPATIBLE WITH HF 
    hf_train_dataset = Dataset.from_dict(train_dataset)
    hf_test_dataset = Dataset.from_dict(test_dataset)

    # GRPO config:

    training_args = GRPOConfig(**config_dict)

    #GRPO
    trainer = GRPOTrainer(
    model="NorseDrunkenSailor/ProtGPT2-with-pad",
    reward_funcs=reward_hydrophobicity,
    args=training_args,
    train_dataset=hf_train_dataset,
    )
    
    trainer.train()



if __name__ == "__main__":
    main()
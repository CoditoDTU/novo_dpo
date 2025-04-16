
# %%
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
import sys
import json
from datasets import Dataset
from datasets import DatasetDict

# GRPO Trainer args

TRAIN_FILENAME = 'amp_dpo_fixed.json'
TRAIN_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TRAIN_FILENAME)

MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")

# GRPO config args

OUTPUT_NAME = 'DPO_protgpt2_9'
LOGGING_STEPS = 1
BETA = 0.1
LEARNING_RATE = 1e-5
ADAM_BETAS = (0.9, 0.999) 
ADAM_EPSILON = 1e-8
N_TRAIN_EPOCHS = 50
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
    'weight_decay': ADAM_DECAY
}



def main():
        
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    with open(TRAIN_DATA_PATH, "r") as f:
        train_dataset = json.load(f)

    hf_dataset = Dataset.from_dict(train_dataset)

    training_args = DPOConfig(**config_dict)

    trainer = DPOTrainer(model = MODEL,
                         args = training_args,
                         train_dataset = hf_dataset,
                         processing_class = TOKENIZER)
    trainer.train()




if __name__ == "__main__":
    main()

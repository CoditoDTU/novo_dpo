
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

TRAIN_FILENAME = 'OXDA_activity_dpo_train.json'
TEST_FILENAME = 'OXDA_activity_dpo_test.json'

TRAIN_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TRAIN_FILENAME)
TEST_DATA_PATH = os.path.join(os.getcwd(), "src", "pyutils", TEST_FILENAME)

MODEL = AutoModelForCausalLM.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")
TOKENIZER = AutoTokenizer.from_pretrained("NorseDrunkenSailor/ProtGPT2-with-pad")


# GRPO config args

OUTPUT_NAME = 'DPO_protgpt2_oxda_4'
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
    'weight_decay': ADAM_DECAY
}



def main():
        

    with open(TRAIN_DATA_PATH, "r") as f:
        train_dataset = json.load(f)

    with open(TEST_DATA_PATH, "r") as f:
        test_dataset = json.load(f)

    hf_train_dataset = Dataset.from_dict(train_dataset)
    hf_test_dataset = Dataset.from_dict(test_dataset)

    training_args = DPOConfig(**config_dict)

    trainer = DPOTrainer(model = MODEL,
                         args = training_args,
                         train_dataset = hf_train_dataset,
                         eval_dataset = hf_test_dataset,
                         processing_class = TOKENIZER)
    trainer.train()
    trainer.save_model()

    eval = trainer.evaluate()
    

    with open(EVAL_JSON_PATH, "w") as json_file:
        json.dump(eval, json_file, indent=4)



if __name__ == "__main__":
    main()

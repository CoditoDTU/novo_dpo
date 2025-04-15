
# %%
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# GRPO Trainer args
MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# GRPO config args

OUTPUT_NAME = 'Qwen2-0.5B-DPO'
LOGGING_STEPS = 1
BETA = 0.1
LEARNING_RATE = 5e-05
ADAM_BETAS = (0.9, 0.999) 
ADAM_EPSILON = 1e-8
N_TRAIN_EPOCHS = 1.0
ADAM_DECAY = 0.0
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
        
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

    training_args = DPOConfig(**config_dict)

    trainer = DPOTrainer(model = MODEL,
                         args = training_args,
                         processing_class = TOKENIZER,
                         train_dataset = train_dataset)
    trainer.train()




if __name__ == "__main__":
    main()

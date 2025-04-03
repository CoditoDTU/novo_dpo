import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from ml_tools import *
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from copy import deepcopy

#HYPER PARAMETERS

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
LERANING_RATE = 1e-6 # Learning rate for optimizerx
#PLOTTING

REWARDS = []
LOSSES = []
GRADIENT_NORMS = []


# Optimizer setup (As mentioned in deepseek math paper)
optimizer = Adam(MODEL.parameters(),
                 lr = LERANING_RATE,
                 betas =(0.9, 0.95),
                 weight_decay = 0.1)


# GRPO training loop

def grpo_training():
    gradient_norms = []
    for iteration in range(I):

        #old_policy_model = deepcopy(MODEL) 
        reference_model = deepcopy(MODEL)

        for step in range(M):
            
            # Update the old policy model πθold ← πθ
            old_policy_model = deepcopy(MODEL)

            # Inference of sequences with model(Sample a batch Db from D)
            generated_sequences = pretrained_inf(model = old_policy_model,
                                                 tokenizer = TOKENIZER,
                                                 condition_tag = INPUT,
                                                 num_seqs = N_SEQS,
                                                 max_length = MAX_LEN)
            
            # Compute tuples and rewards using the oracles
            seq_tuples = parsing_seqs_tuples(generated_sequences)

            
            seq_list_pI = calculate_physicochemical(tuples = seq_tuples,
                                                    physico_param = 'pI')
            # Compute Rewards
            rewards = compute_rewards(reference = REFERENCE_VALUE,
                                      metric_list = seq_list_pI)
            print(f"rewards: {rewards}")
            REWARDS.append(list(rewards))
            
            # Compute advantages:
            advantages = compute_advantage(rewards = rewards)

            

            # Loss calculation:
            for grpo_iter in range(MU):
                # Compute the token LL of the models

                # Policy model
                policy_length, policy_token_nll = compute_metrics_multiple(mode = 'token',
                                                                        seq_list = generated_sequences,
                                                                        model = MODEL,
                                                                        tokenizer = TOKENIZER,
                                                                        gradient = 'yes')
                
                # Old policy model
                _, old_policy_token_nll = compute_metrics_multiple(mode = 'token',
                                                                seq_list = generated_sequences,
                                                                model = old_policy_model,
                                                                tokenizer = TOKENIZER,
                                                                gradient = 'no')
                
                # Reference model
                _, reference_token_nll = compute_metrics_multiple(mode = 'token',
                                                                seq_list = generated_sequences,
                                                                model = reference_model,
                                                                tokenizer = TOKENIZER,
                                                                gradient = 'no')

                # Compute loss and update policy
                loss = compute_Jgrpo(policy_nll = policy_token_nll,
                                 old_policy_nll = old_policy_token_nll,
                                 ref_nll = reference_token_nll,
                                 advantages = advantages,
                                 beta = BETA,
                                 epsilon = EPSILON,
                                 lengths_tensor = policy_length)
                
                optimizer.zero_grad()
                loss.backward()

                # ========== GRADIENT NORM CALCULATION ==========
                # total_norm, layer_norms = calculate_gradient_norm(MODEL, norm_type=2)
                # gradient_norms.append(total_norm)

                # # Optional: Print detailed layer norms
                # print(f"\nGlobal Gradient Norm: {total_norm:.4f}")
                # for name, norm in layer_norms.items():
                #     print(f"{name}: {norm:.4f}")


                optimizer.step()

            print(f"Loss : {loss}")
    return MODEL


trained_model = grpo_training()
# After trained_model = grpo_training()
# plt.figure(figsize=(10, 5))
# plt.plot(gradient_norms)
# plt.title("Gradient Norms During Training")
# plt.xlabel("Update Step")
# plt.ylabel("L2 Norm")
# plt.savefig("gradient_norms.png")
# plt.show()

# # Plotting the rewards
# plt.figure(figsize=(10, 6))  # Set the figure size
# for i, rewards in enumerate(REWARDS):
#     plt.plot(rewards, label=f'Step {i + 1}')

# plt.title('Rewards Over Training Steps')
# plt.xlabel('Sequence Number')
# plt.ylabel('Reward Value')
# plt.legend()  # Add a legend to distinguish steps
# plt.grid(True)  # Add a grid for better readability

# # Save the plot as a file
# plt.savefig('rewards_plot.png', dpi=300, bbox_inches='tight')  # Save as PNG file

# plt.show()  # Display the plot
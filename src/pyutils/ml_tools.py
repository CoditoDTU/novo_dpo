import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import re
import sys
import numpy as np

#Inference of pretrained model
def pretrained_inf(model, tokenizer, condition_tag :str, num_seqs :int, max_length:int ): # for HF models v0.0.1
    '''
    Function to run HF model inference. Returns a list of generated sequences based on a condition tag or sequence
    model: recieves model object for inference when training 
    model_name: Model name from the pretrain model in huggingface EX: model_name = "nferruz/ProtGPT2"
    condition_tag: String primer for the generation of text. Always must start with <|endoftext|>.
    num_seqs: Number of sequences you want to generate based on your condition
    max_length: Maximun length desired for generated sequences

    '''
    #Check GPU availability 
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')



    # Tokenization
    input_tokens = tokenizer.encode(condition_tag,
                                    return_tensors = "pt").to(device)
    
    
    # Inference
    output = model.generate(input_tokens,
                            max_length = max_length,
                            do_sample = True,
                            top_k = 950,
                            repetition_penalty = 1.2,
                            num_return_sequences = num_seqs,
                            eos_token_id = 0 )
    
    #Decoding 
    seq_output = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

    return seq_output

# Metrics function
def compute_metrics_single(mode:str, sequence:str, model, tokenizer, gradient = 'no'):
    '''
    Calculates PPL and Nll out of the forward pass of the model.

    mode: str that specifies which metric to calculate. it can only be ppl or nll
    sequence: generated sequence to evaluate and to get ppl or nll from 
    model: model object from HF 
    tokenizer: tokenizer object from HF
    '''
    #Check GPU availability 
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(device)

    if gradient == 'no':
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

    elif gradient == 'yes':
        outputs = model(input_ids, labels=input_ids)
    
    loss, logits = outputs[:2]

    per_token_ll = logits
   #nll = loss.item()
    ppl = math.exp(loss)

    if mode == 'ppl':
        output = ppl
    elif mode == 'nll':
        output = "nono"
    elif mode == 'token':
        output = per_token_ll
    return output



def compute_metrics_multiple(mode: str, seq_list: list, model, tokenizer,gradient: str):

    ppl_list = list(map(lambda x: compute_metrics_single(sequence = x, 
                                                  model = model,
                                                  tokenizer = tokenizer,
                                                  mode = mode,
                                                  gradient = gradient), seq_list))
    
    # Extract the lengths (dimension 1) of each tensor in ppl_list
    lengths = torch.tensor([p.shape[1] for p in ppl_list], dtype=torch.long)

    # print(f'dimension of ppl list 0 : {ppl_list[0].shape}\n',
    #       f'dimension of ppl list 1 : {ppl_list[1].shape}\n',
    #       f'lengths of sequences encoded: {lengths}')
    ordered_list = torch.cat(ppl_list, dim=1).view(-1)
    return lengths, ordered_list #torch.tensor(ordered_list)


def parsing_seqs_single(seq:str):
    '''
    remove special characters from ProtGPT2 parsing of AA sequences 
    '''
     # Remove all newline characters
    cleaned_string = seq.replace('\n', '')
    # Remove all occurrences of '<|endoftext|>'
    cleaned_string = cleaned_string.replace('<|endoftext|>', '')

    return cleaned_string


def parsing_seqs_tuples(sequences:list):

    norm_seqs = list(map(lambda x: parsing_seqs_single(x), sequences))

    tuple_seqs = list(zip(sequences, norm_seqs))

    return tuple_seqs


# def calculate_physicochemical(tuples, physico_param=None):
#     # Define the hydrophobicity dictionary (Kyte-Doolittle scale)
#     hydrophobicity = {
#         'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
#         'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
#         'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
#         'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
#     }
    
#     # Define pKa values for ionizable groups
#     pka = {
#         'N_term': 8.6, 'C_term': 3.1,
#         'D': 3.9, 'E': 4.3, 'H': 6.0, 'C': 8.4,
#         'Y': 10.1, 'K': 10.8, 'R': 12.5
#     }
    
#     sequences = [t[1] for t in tuples]
#     #sequences = [sequence fo sequence in sequences]
#     if physico_param == 'phob':
#         return [sum(hydrophobicity.get(aa, 0.0) for aa in seq) for seq in sequences]
#     elif physico_param == 'pI':
#         results = []
#         for seq in sequences:
#             counts = {
#                 'D': seq.count('D'), 'E': seq.count('E'),
#                 'H': seq.count('H'), 'C': seq.count('C'),
#                 'Y': seq.count('Y'), 'K': seq.count('K'),
#                 'R': seq.count('R'), 'N_term': 1, 'C_term': 1
#             }
            
#             def net_charge(pH):
#                 charge = 0.0
#                 # N-terminus contribution
#                 charge += 1.0 / (1 + 10 ** (pH - pka['N_term']))
#                 # C-terminus contribution
#                 charge -= 1.0 / (1 + 10 ** (pka['C_term'] - pH))
#                 # Asp (D)
#                 charge -= counts['D'] / (1 + 10 ** (pka['D'] - pH))
#                 # Glu (E)
#                 charge -= counts['E'] / (1 + 10 ** (pka['E'] - pH))
#                 # His (H)
#                 charge += counts['H'] / (1 + 10 ** (pH - pka['H']))
#                 # Cys (C)
#                 charge -= counts['C'] / (1 + 10 ** (pka['C'] - pH))
#                 # Tyr (Y)
#                 charge -= counts['Y'] / (1 + 10 ** (pka['Y'] - pH))
#                 # Lys (K)
#                 charge += counts['K'] / (1 + 10 ** (pH - pka['K']))
#                 # Arg (R)
#                 charge += counts['R'] / (1 + 10 ** (pH - pka['R']))
#                 return charge
            
#             # Binary search between pH 0 and 14
#             low, high = 0.0, 14.0
#             for _ in range(100):
#                 mid = (low + high) / 2
#                 if net_charge(mid) > 0:
#                     low = mid
#                 else:
#                     high = mid
#             results.append(round((low + high) / 2, 2))
#         return results
#     else:
#         raise ValueError("Unsupported physicochemical parameter")
    
def reward_hydrophobicity(completions, **kwargs):
    # Kyte-Doolittle hydrophobicity scale
    hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }

    rewards = []
    for seq in completions:
        total = -abs(20 - sum(hydrophobicity.get(aa, 0.0) for aa in seq))
        rewards.append(total)
    return rewards
 
 
def compute_rewards(reference: int, metric_list: list):
    '''
    Compute reward function. Based on the results of oracle, it will take those values and give a reward to a list of scores.
    reference: int value representing the desired propertie Ex: Target pH, 
    '''
    reward = -abs(np.array(metric_list) - reference) ## Operation to calculate reward based on biochemical propertie

    return reward


def compute_kl_estimator(policy_nll, ref_nll):

    # policy_probs = torch.softmax(policy_nll, dim=-1)
    # ref_probs = torch.softmax(ref_nll, dim=-1)
    

        
    ratio = torch.exp(ref_nll-policy_nll)

    log_term = ref_nll - policy_nll

    kl_divergence = ratio - log_term - 1 # Should ALWAYS BE POSITIVE
    return kl_divergence

def compute_advantage(rewards):

    mu = np.mean(rewards)
    sigma2 = np.std(rewards)

    advantages = (rewards - mu)/sigma2

    return torch.tensor(advantages)



def compute_Jgrpo(policy_nll: torch.tensor, old_policy_nll: torch.tensor, advantages: torch.tensor, ref_nll : torch.tensor,lengths_tensor: torch.tensor, beta = 0.1, epsilon  = 0.01,  vocab_size =50257 ):
    '''
    Compute the J_GRPO = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t] - β * KL(policy || ref)
    policy_nll: per token log-likelihood of the policy model (logits) Dimension(1*token_length*vocab_size)
    old_policy_nll: NLL of previous instance of the policy model Dimension(1*token_length*vocab_size)
    advantages: tensor of Advantages of each sample from the group of samples: Dimension(n_sequence)
    lengths_tensor: Tensor with the variable lengths of generated sequencer: Dimension (N_sequences)
    vocab_size: size of vocabulary. Used for broadcasting advantages from (n_sequence) --> (1*lengths_tensor*vocab_size)
    epsilon: Hyper_parameter
    length_tensor: torch tensor with all the lengths of generated sequences, used for broadcasting the advantages.
    '''

    # policy_probs = torch.softmax(policy_nll, dim=-1)
    # ref_probs = torch.softmax(old_policy_nll, dim=-1)

    # ratio2 = torch.exp(policy_probs - ref_probs)
    ratio = torch.exp(policy_nll - old_policy_nll)
    print(f"ratios: {ratio}")

    # Clipped surrogated objective
    clipped_ratio = torch.clamp(ratio,
                                1 - epsilon,
                                1 + epsilon)


    advantages_extended = advantages.repeat_interleave(lengths_tensor)
    advantages_flattened = advantages_extended.repeat_interleave(vocab_size)

    surrogate = torch.min(ratio * advantages_flattened,
                          clipped_ratio * advantages_flattened)

    # KLD penalty 
    kl_penalty = compute_kl_estimator(policy_nll = policy_nll,
                                      ref_nll = ref_nll)


    # j_GRPO
    grpo_objective = (surrogate - beta * kl_penalty).mean()

    return -1*grpo_objective


# def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
#         # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
#         logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
#         logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

#         input_ids = input_ids[:, -logits_to_keep:]
#         # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
#         # See https://github.com/huggingface/trl/issues/2770
#         logits = logits[:, -logits_to_keep:]
#         return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

def calculate_gradient_norm(model, norm_type=2):
    """Compute gradient norms for monitoring.
    
    Args:
        model: PyTorch model
        norm_type: Type of norm (2 for L2, 'max' for infinity norm)
    
    Returns:
        total_norm: Global gradient norm
        layer_norms: Dict of layer-wise gradient norms
    """
    total_norm = 0.0
    layer_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if norm_type == 'max':
                param_norm = param.grad.data.abs().max().item()
            else:
                param_norm = param.grad.data.norm(norm_type).item()
            
            layer_norms[name] = param_norm
            total_norm += param_norm ** norm_type if norm_type != 'max' else param_norm
    
    if norm_type != 'max':
        total_norm = total_norm ** (1.0 / norm_type)
    
    return total_norm, layer_norms

## MAIN FUNCTION FOR TESTING 
def main():

    #HYPER PARAMETERS
    INPUT = "<|endoftext|>MVKVLAVLYDGGEHAKQVPG"
    N_SEQS = 5
    MAX_LEN = 10
    MODEL_NAME = "nferruz/ProtGPT2"
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    FMODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # GRPO HYPERPARAMETERS
    EPSILON = 0.01
    BETA = 0.1
    
    # Inference for sequences
    generated_sequences = pretrained_inf(model = MODEL,
                                         tokenizer = TOKENIZER,
                                        condition_tag = INPUT, 
                                        num_seqs = N_SEQS,
                                        max_length = MAX_LEN)
    
    total_norm, layer_norms = calculate_gradient_norm(MODEL, norm_type=2)

    # Data generation generation
    gen_seq_tuples = parsing_seqs_tuples(sequences = generated_sequences)
    
    
    policy_length, policy_token_ll = compute_metrics_multiple(seq_list = generated_sequences,
                                           mode='token',
                                           model = MODEL,
                                           tokenizer = TOKENIZER,
                                           gradient = 'yes')

    _, reference_token_ll = compute_metrics_multiple(seq_list = generated_sequences,
                                           mode='token',
                                           model = FMODEL,
                                           tokenizer = TOKENIZER,
                                           gradient = 'no')
    
    
    # NLL from Reference model
    _, seq_list_nll_fmodel= compute_metrics_multiple(seq_list = generated_sequences,
                                           mode='token',
                                           model = FMODEL,
                                           tokenizer = TOKENIZER,
                                           gradient = 'no')
    
    
    # Oracle properties calculation(metric lists)
    seq_list_pI = calculate_physicochemical(gen_seq_tuples,
                                             physico_param='pI') # Isoelectric point
    
    seq_list_phob = calculate_physicochemical(gen_seq_tuples,
                                               physico_param='phob') # hydrophobicity

    # Compute Rewards
    rewards = compute_rewards(reference = 3.0, 
                              metric_list = seq_list_pI)
    

    # Compute Advantage:

    advantages = compute_advantage(rewards)

    # JGRPO 
    loss = compute_Jgrpo(policy_nll = policy_token_ll,
                         old_policy_nll = reference_token_ll,
                         ref_nll = seq_list_nll_fmodel,
                         advantages = advantages,
                         beta = BETA,
                         epsilon = EPSILON,
                         lengths_tensor = policy_length)

    loss.backward()
    
    
    print(f"Generated sequences tuples {generated_sequences}\n",
          f"NLL{policy_token_ll}\n", 
          f"reference token ll {reference_token_ll}\n",
          f"pI:{seq_list_pI}\n",
          f"hydrophobicity:{seq_list_phob}\n",
          f"Reward:{rewards}\n",
          f"Advantages:{advantages}\n",
          f"Jgrpo:{loss}\n")


if __name__ == "__main__":
    main()
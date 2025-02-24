import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import re

#Inference of pretrained model
def pretrained_inf(model_name :str, condition_tag :str, num_seqs :int, max_length:int ): # for HF models v0.0.1
    '''
    Function to run HF model inference. Returns a list of generated sequences based on a condition tag or sequence 
    model_name: Model name from the pretrain model in huggingface EX: model_name = "nferruz/ProtGPT2"
    condition_tag: String primer for the generation of text. Always must start with <|endoftext|>.
    num_seqs: Number of sequences you want to generate based on your condition
    max_length: Maximun length desired for generated sequences

    '''
    #Check GPU availability 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # call model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    #tokenization
    input_tokens = tokenizer.encode(condition_tag, return_tensors = "pt").to(device)
    
    
    #Inference
    output = model.generate(input_tokens, max_length = max_length, do_sample =True, top_k = 950, repetition_penalty=1.2, num_return_sequences=num_seqs, eos_token_id=0 )

    #Decoding 
    seq_output = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

    return seq_output

# Metrics function
def compute_metrics_single(mode:str, sequence:str, model, tokenizer):
    '''
    Calculates PPL and Nll out of the forward pass of the model.

    mode: str that specifies which metric to calculate. it can only be ppl or nll
    sequence: generated sequence to evaluate and to get ppl or nll from 
    model: model object from HF 
    tokenizer: tokenizer object from HF
    '''
    #Check GPU availability 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0) 
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    nll = loss.item()
    ppl = math.exp(loss)

    if mode == "ppl":
        output = ppl
    elif mode == 'nll':
        output = nll

    return output

# Metrics function for multiple sequences
def compute_metrics_multiple(mode: str, seq_list: list, model, tokenizer):
    ppl_list = list(map(lambda x: compute_metrics_single(sequence = x, 
                                                  model = model,
                                                  tokenizer = tokenizer,
                                                  mode = mode), seq_list))
    
    ordered_list = sorted(ppl_list)

    return ordered_list

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



## MAIN FUNCTION FOR TESTING 
def main():

    #HYPER PARAMETERS
    INPUT = "<|endoftext|>MVKVLAVLYDGGEHAKQVPG"
    N_SEQS = 10
    MAX_LEN = 370
    MODEL_NAME = "nferruz/ProtGPT2"
    MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    

    # Inference for sequences
    generated_sequences = pretrained_inf(model_name = MODEL_NAME , 
                                        condition_tag = INPUT, 
                                        num_seqs = N_SEQS,
                                        max_length = MAX_LEN)
    
    gen_seq_tuples = parsing_seqs_tuples(sequences = generated_sequences)
    
    print(gen_seq_tuples)



    raise NotImplementedError
    # Get Metrics
    seq_list_ppl= compute_metrics_multiple(seq_list = generated_sequences,
                                            mode ='ppl',
                                            model = MODEL,
                                            tokenizer = TOKENIZER)
    
    seq_list_nll= compute_metrics_multiple(seq_list = generated_sequences,
                                           mode='nll',
                                           model = MODEL,
                                           tokenizer = TOKENIZER)
    
    
    
    # print(f"Generated sequences tuples {generated_sequences}",
    #       f"NLL{seq_list_nll}", 
    #       f"PPL{seq_list_ppl}")


if __name__ == "__main__":
    main()
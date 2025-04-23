
# %%
import pandas as pd
import numpy as np
import os 
import sys
import json
import random
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
from src.pyutils.ml_tools import  reward_hydrophobicity
from src.pyutils.data_utils import *
# %%
FILENAME = 'antimicrobial_peptides_dataset_raw.tsv'
DICT_NAME = 'amp_dpo_fixed.json'
PATH = os.path.join(os.getcwd(), 'data','raw', FILENAME)
OUTPUT_PATH = os.path.join(os.getcwd(), 'data','processed', DICT_NAME)





def main():

    df = pd.read_csv(filepath_or_buffer = PATH,
                     sep = '\t')
    
    dict = parse_sequences_to_preference_dict_fixedn(df = df)

    save_dataset(dictionaries = dict,
                 output_path = OUTPUT_PATH )
    
    raise NotImplementedError
    sequences = df[['Sequence']].copy()

    generated_sequences = list(map(lambda x: generate_random_aa_sequence(x),
                                   sequences['Sequence'].str.len() ))
    
    sequences['generated_sequences'] = generated_sequences
    sequences['reward_real'] = reward_hydrophobicity(completions = sequences['Sequence'])
    sequences['reward_generated'] = reward_hydrophobicity(completions = sequences['generated_sequences'])
    print(sequences.head())

    sequences.to_csv(path_or_buf = "")

main()
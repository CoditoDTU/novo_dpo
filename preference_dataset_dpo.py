
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
import matplotlib.pyplot as plt 

# %%

# Ideas to automate
'''
Variable for changing filename and path
Variable for selecting column metric
Variable to assign a pattern or apend M at the beggining
'''


# %%
FILENAME = 'gh114_2.csv'
PATH = os.path.join(os.getcwd(), 'data','raw', FILENAME)

DICT_NAME_TRAIN = 'gh114_2_train.json'
# DICT_NAME_VALIDATE = 'ppat_dpo_val.json'
# DICT_NAME_TEST = 'ppat_dpo_test.json'


OUTPUT_PATH_TRAIN = os.path.join(os.getcwd(), 'data','processed', DICT_NAME_TRAIN)
# OUTPUT_PATH_VAL = os.path.join(os.getcwd(), 'data','processed', DICT_NAME_VALIDATE)
# OUTPUT_PATH_TEST = os.path.join(os.getcwd(), 'data','processed', DICT_NAME_TEST)

# %%

def main():

    df = pd.read_csv(filepath_or_buffer = PATH )
    # prefix = 'HSQKR'
    # df_prefix = filter_rows_by_pattern(df = df,
    #                                     column_name ='mutated_sequence',
    #                                     pattern = prefix)

    df_sorted = df.sort_values(by = 'target_reg', 
                                        ascending = False)
    
    df_test = df_sorted

    # PREFERENCE DATASET

    y = df_test['target_reg'].to_list()
    print(y)

    sequences = df_test['sequence'].to_list()
    epsilon = 0.3

    pairs = construct_pairs(Yvec = y, 
                            epsilon = epsilon)
    
    # dictionary = format_string_pairs(strings = sequences,
    #                                  index_pairs = pairs,
    #                                  N_characters = 4)
    # save_dataset(dictionaries=dictionary,
    #              output_path= OUTPUT_PATH_TRAIN)
    
    #raise NotImplementedError
    # train_pairs, temp_pairs = train_test_split(pairs,
    #                                            test_size = 0.3,
    #                                            random_state = 42)

    # val_pairs, test_pairs = train_test_split(pairs,
    #                                          test_size = 0.5,
    #                                          random_state = 42)
    
    train_dict = format_string_pairs(strings = sequences,
                                     index_pairs = pairs,
                                     N_characters = None)
    

    print(train_dict)
    # val_dict = format_string_pairs(strings = sequences,
    #                                  index_pairs = val_pairs,
    #                                  N_characters = None)

    # test_dict = format_string_pairs(strings = sequences,
    #                                  index_pairs = test_pairs,
    #                                  N_characters = None)
    
    save_dataset(dictionaries=train_dict,
                 output_path= OUTPUT_PATH_TRAIN)
    
    # save_dataset(dictionaries=test_dict,
    #              output_path= OUTPUT_PATH_TEST)
    
    # save_dataset(dictionaries=val_dict,
    #              output_path= OUTPUT_PATH_VAL)
    
    
    
    



main()
# %%

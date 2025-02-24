from pyutils.data_utils import proteinDataset, TaxonIdSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import numpy as np 


# Hyperparameters

BATCH_SIZE = 8
DATA_FILE = '../data/test100.parquet'
OUTPUT_DIR = "../data/outputs/"
MODEL = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
REP_LAYER = 6 #ensure it matches the model
COLUMNS = ['ogt', 'taxid', 'sequence']


## DATA LOADING 

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_data = pd.read_parquet(DATA_FILE, columns = COLUMNS)
dataset  = proteinDataset(raw_data)
sampler =  TaxonIdSampler(dataset= dataset, batch_size = BATCH_SIZE, shuffle = True)
dataloader = DataLoader(dataset, batch_sampler= sampler, collate_fn =lambda x: x, shuffle = False)

dataset_size = len(dataloader)

test_batch = next(iter(dataloader))

print(test_batch)
raise NotImplementedError
for n, batch in enumerate(dataloader):
    print(batch)
    if n >=5:
        break
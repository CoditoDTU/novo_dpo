import numpy as np
import pandas as pd
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler


## DATASET FOR PROTEINS:


class proteinDataset(Dataset):

    def __init__(self, data):
        
        self.data = data
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):

        item = self.data.iloc[index]

        sequence = item['sequence']
        length = len(item['sequence']) # torch.tensor(item['length'])
        taxon_id = item['taxid']
        tg = item['ogt']


        return {
            'sequence': sequence,  # Return sequence as a string (you could modify this later)
            'taxon_id': taxon_id,
            'temp_g': tg,
            'length': length
        }
    
## SAMPLER FOR DATA:

class TaxonIdSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size, length_bin_size=5, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_bin_size = length_bin_size
        self.shuffle = shuffle

        # Group sample indices by taxonId
        self.taxon_length_bins = defaultdict(lambda: defaultdict(list))

        for idx, sample in enumerate(dataset):
            taxon_id = sample['taxon_id']
            sequence_length = sample['length']
            length_bin = (sequence_length // length_bin_size) * length_bin_size  # integer division to know in which bucket the sequence is

            # Ensure that length_bin is properly initialized
            self.taxon_length_bins[taxon_id][length_bin].append(idx)

        '''
        structure of self.taxon_length_bins:

        {
            taxon_id_1: {
                length_bin_1: [sample_idx_1, sample_idx_2, ...],
                length_bin_2: [sample_idx_3, sample_idx_4, ...],
                ...
            },
            taxon_id_2: {
                length_bin_3: [sample_idx_5, sample_idx_6, ...],
                ...
            },
        }
        '''
        
        # Prepare batches based on taxon groups
        self.batches = []

        for taxon, length_bins in self.taxon_length_bins.items():
            for length_bin, indices in length_bins.items():
                if self.shuffle:
                    random.seed(42)
                    random.shuffle(indices)  # Shuffle the indices if needed
                for i in range(0, len(indices), batch_size):
                    self.batches.append(indices[i:i + batch_size])

        # Shuffle the batches if needed
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

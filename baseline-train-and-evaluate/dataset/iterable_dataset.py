import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from Bio import SeqIO
import os
from tqdm import tqdm


NUCLEOTIDE_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4, '<PAD>': 5, '<MASK>': 6}
VOCAB_SIZE = len(NUCLEOTIDE_VOCAB)
PAD_IDX = NUCLEOTIDE_VOCAB['<PAD>']
MASK_IDX = NUCLEOTIDE_VOCAB['<MASK>']

DATASET_ROOT = "/home/ec2-user/rna1/baseline_downstream/datasets-after-filtering/"
DATASET_FILENAME = "threshold_0.95_1-sp_mmseqs_clustered_split_train.fasta"
#DATASET_FILENAME = "100000_rnacentral.fasta"
DATA_PATH = os.path.join(DATASET_ROOT, DATASET_FILENAME)

class CharTokenizer:
    """Tokenizer for nucleotide sequences."""
    def __init__(self, vocab):
        self.vocab = vocab
    def tokenize(self, sequence):
        # Return vocab index for each nucleotide, defaults to N if unknown nucleotide type
        return [self.vocab.get(i, self.vocab['N']) for i in sequence.upper()]

class MaskedRNAIterableDataset(IterableDataset):
    """Dataset for masked RNA sequences for MLM tasks."""
    def __init__(self, data_path, tokenizer, vocab, max_length=512, mask_prob=0.15, seed=42, padding=False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        # Seed for reproducibility
        self.seed = seed
    
        # Set mask and pad token ids
        self.mask_token_id = vocab['<MASK>']
        self.pad_idx = vocab['<PAD>']
        self.mask_prob = mask_prob
        
    def __iter__(self):
        with open(self.data_path, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                sequence = str(record.seq)
                tokens = self._tokenize(sequence, self.tokenizer, self.max_length, padding=self.padding, padding_idx=self.pad_idx)
                att_mask = [0] * len(tokens) + [1] * (self.max_length - len(tokens))
                masked_sequence, label = self._create_masked_sample(tokens, self.mask_token_id, self.pad_idx, self.mask_prob, self.seed)
                yield torch.tensor(masked_sequence), torch.tensor(label), torch.tensor(att_mask)
        
    @staticmethod
    def _create_masked_sample(sequence, mask_token_id, pad_idx, mask_prob=0.15, seed=42):
        """Create masked inputs and labels for nucleotide sequence."""
        random.seed(seed)
        
        masked_sequence = sequence.copy()
        label = [-100] * len(sequence)  # -100 will be ignored in loss calculation
        # Set the random seed for reproducibility
        
        for i in range(len(sequence)):
            # Don't mask padding tokens DONE
            if sequence[i] == pad_idx:
                continue
            if random.random() < mask_prob:
                label[i] = sequence[i]  # Original token becomes the label
                
                mask_type = random.random()
                if mask_type < 0.8:
                    masked_sequence[i] = mask_token_id  # 80%: mask token
                elif mask_type < 0.9:
                    masked_sequence[i] = random.randint(0, 4)  # 10%: random nucleotide
                # 10%: leave unchanged
        return masked_sequence, label
        
            
    @staticmethod
    def _tokenize(sequence, tokenizer, max_length=1024, padding=False, padding_idx=None):
        """Tokenizes the sequence into chunks of the given length, using sliding window to avoid hard splitting in important regions."""
        tokens = tokenizer.tokenize(sequence)  # Always tokenize the string first
        #print(f"sequence: {sequence}, tokens: {tokens}")
        if padding:
            if padding_idx is None:
                raise ValueError("padding_idx must be provided if padding is True")
            tokens = tokens + [padding_idx] * (max_length - len(tokens))
        return tokens

if __name__ == "__main__":
    tokenizer = CharTokenizer(NUCLEOTIDE_VOCAB)
    dataset = MaskedRNAIterableDataset(DATA_PATH, tokenizer, NUCLEOTIDE_VOCAB, padding=True, max_length=1024)
    loader = DataLoader(dataset, batch_size=256)
    # Only print the first couple of batches
    
    for i, batch in enumerate(loader):
        print(f"batch {i}: {batch[0]}")


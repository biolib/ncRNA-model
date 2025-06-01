import torch
from torch.utils.data import Dataset
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
DATASET_FILENAME = "threshold_0.95_1-sp_mmseqs_clustered.fasta"
#DATASET_FILENAME = "100000_rnacentral.fasta"
DATA_PATH = os.path.join(DATASET_ROOT, DATASET_FILENAME)

class CharTokenizer:
    """Tokenizer for nucleotide sequences."""
    def __init__(self, vocab):
        self.vocab = vocab
    def tokenize(self, sequence):
        # Return vocab index for each nucleotide, defaults to N if unknown nucleotide type
        return [self.vocab.get(i, self.vocab['N']) for i in sequence.upper()]

class MaskedRNADataset(Dataset):
    """Dataset for masked RNA sequences for MLM tasks."""
    def __init__(self, data_path, tokenizer, vocab, max_length=512, window_size=None, stride=None, mask_prob=0.15, seed=42, train=True, train_split=0.95):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Seed for reproducibility
        self.seed = seed
        
        # Whether to use training or validation set
        self.train = train
        self.train_split = train_split
        # Sliding window parameters if the sequence is longer than max_length
        self.window_size = window_size
        self.stride = stride
        
        # Load data from FASTA file
        self._load_data()
    
        # Set mask and pad token ids
        self.mask_token_id = vocab['<MASK>']
        self.pad_idx = vocab['<PAD>']
        self.mask_prob = mask_prob
        
        # Create masked samples
        self._create_dataset()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), torch.tensor(self.att_mask[idx])
    
    
    def _create_dataset(self):
        """Splits the sequences into even length chunks, tokenizes them and masks them."""
        # self.n_samples_from_sequence if for DEBUG
        
        if self.train:
            tokens, self.n_samples_from_sequence = self._batch_tokenize(self.data[:int(len(self.data) * self.train_split)], self.tokenizer, self.max_length, self.window_size, self.stride)
        else:
            tokens, self.n_samples_from_sequence = self._batch_tokenize(self.data[int(len(self.data) * self.train_split):], self.tokenizer, self.max_length, self.window_size, self.stride)
        att_mask = [[1] * len(token) + [0] * (self.max_length - len(token)) for token in tokens]
        padded_tokens = [token + [self.pad_idx] * (self.max_length - len(token)) for token in tokens]
        
        self.X, self.y, self.att_mask = *self._create_masked_samples(padded_tokens, self.mask_token_id, self.pad_idx, self.mask_prob, self.seed), att_mask

    def _load_data(self):
        """Loads the data from the FASTA file."""
        sequences = []
        for record in tqdm(SeqIO.parse(self.data_path, "fasta"), desc="Loading sequences", unit="seq"):
            sequences.append(str(record.seq))
            
        random.seed(self.seed)
        random.shuffle(sequences)
        
        self.data = sequences
        
    @staticmethod
    def _create_masked_samples(sequences, mask_token_id, pad_idx, mask_prob=0.15, seed=42):
        """Create masked inputs and labels for nucleotide sequence."""
        random.seed(seed)
        
        masked_sequences = []
        labels = []
        
        # TODO: Make this more efficient with matrices
        
        for sequence in tqdm(sequences, desc="Creating masked samples", unit="seq"):
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
                    
            masked_sequences.append(masked_sequence)
            labels.append(label)
        return masked_sequences, labels
        
            
    @staticmethod
    def _batch_tokenize(sequences, tokenizer, max_length=512, window_size=None, stride=None):
        """Tokenizes the sequences into chunks of the given length, using sliding window to avoid hard splitting in important regions."""
        if window_size is None:
            window_size = max_length
        if stride is None:
            stride = window_size // 2
        
        tokenized_sequences = []
        n_samples_from_sequence = []
        
        for sequence in tqdm(sequences, desc="Tokenizing sequences", unit="seq"):
            if len(sequence) > max_length:
                curr_seqs = []
                #print(f"len(sequence): {len(sequence)}")
                curr_stride = min(stride, len(sequence) - max_length)
                #print(f"curr_stride: {curr_stride}")
                for i in range(0, len(sequence) - max_length + 1, curr_stride):
                    #print(f"chunk: {i,i+max_length}")
                    curr_seqs.append(tokenizer.tokenize(sequence[i:i+max_length]))
                    
                # Add the last chunk (which may be shorter than max_length)
                if (len(sequence) - max_length) % curr_stride != 0 or len(sequence) < max_length:
                    curr_seqs.append(tokenizer.tokenize(sequence[-max_length:]))
                #print(f"curr_seqs: for {sequence} is {curr_seqs}")
                tokenized_sequences.extend(curr_seqs)
                n_samples_from_sequence.append(len(curr_seqs))
            else:
                tokenized_sequences.append(tokenizer.tokenize(sequence))
                n_samples_from_sequence.append(1)
        
        return tokenized_sequences, n_samples_from_sequence
    

def plot_sequence_length_distribution(sequences, output_path="sequence_length_distribution.png", bins=100):
    """
    Plot the distribution of sequence lengths.
    
    Args:
        sequences: List of sequences
        output_path: Path to save the plot
        bins: Number of bins for the histogram
    """
    lengths = [len(seq) for seq in sequences]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sequence Lengths')
    plt.grid(axis='y', alpha=0.75)
    
    # Add statistics
    plt.axvline(np.mean(lengths), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(lengths):.1f}')
    plt.axvline(np.median(lengths), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(lengths):.1f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return lengths

def plot_nucleotide_distribution(sequences, output_path="nucleotide_distribution.png"):
    """
    Plot the distribution of nucleotides in the sequences.
    
    Args:
        sequences: List of sequences
        output_path: Path to save the plot
    """
    # Count nucleotides
    nucleotide_counts = Counter()
    for seq in sequences:
        nucleotide_counts.update(seq.upper())
    
    # Filter out non-nucleotide characters
    nucleotides = ['A', 'C', 'G', 'U', 'T', 'N']
    counts = [nucleotide_counts.get(n, 0) for n in nucleotides]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(nucleotides, counts, color=['green', 'blue', 'orange', 'red', 'purple', 'gray'])
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:,}', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Nucleotide')
    plt.ylabel('Count')
    plt.title('Distribution of Nucleotides')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return dict(nucleotide_counts)

def read_fasta(fasta_path):
    """
    Read sequences from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        List of sequences
    """
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def plot_token_distribution(token_sequences, vocab, output_path="token_distribution.png"):
    """
    Plot the distribution of tokens in the dataset.X (tokenized, padded, masked sequences).
    Args:
        token_sequences: List of tokenized sequences (list of lists of ints)
        vocab: The vocabulary dict used for tokenization
        output_path: Path to save the plot
    """
    # Flatten all tokens into a single list
    all_tokens = [token for seq in token_sequences for token in seq]
    token_counts = Counter(all_tokens)

    # Map indices to labels, using 'T' for 3 (not 'U')
    index_to_label = {v: k for k, v in vocab.items()}
    # Overwrite 3 to 'T' (since there are no 'U' values)
    index_to_label[3] = 'T'
    # Ensure order: A, C, G, T, N, <PAD>, <MASK>
    ordered_indices = [vocab['A'], vocab['C'], vocab['G'], vocab['T'], vocab['N'], vocab['<PAD>'], vocab['<MASK>']]
    labels = [index_to_label[i] for i in ordered_indices]
    counts = [token_counts.get(i, 0) for i in ordered_indices]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['green', 'blue', 'orange', 'purple', 'gray', 'black', 'red'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:,}', ha='center', va='bottom', rotation=0)

    plt.xlabel('Token')
    plt.ylabel('Count')
    plt.title('Distribution of Tokens in Training Data (dataset.X)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return dict(zip(labels, counts))

def plot_label_distribution(label_sequences, vocab, output_path="label_distribution.png"):
    """
    Plot the distribution of labels in the dataset.y (masked positions, -100 ignored).
    Args:
        label_sequences: List of label sequences (list of lists of ints)
        vocab: The vocabulary dict used for tokenization
        output_path: Path to save the plot
    """
    # Flatten all labels into a single list, ignoring -100
    all_labels = [label for seq in label_sequences for label in seq if label != -100]
    label_counts = Counter(all_labels)

    # Map indices to labels, using 'T' for 3 (not 'U')
    index_to_label = {v: k for k, v in vocab.items()}
    index_to_label[3] = 'T'
    # Only plot nucleotide and special tokens that could be masked (A, C, G, T, N, <PAD>, <MASK>)
    ordered_indices = [vocab['A'], vocab['C'], vocab['G'], vocab['T'], vocab['N'], vocab['<PAD>'], vocab['<MASK>']]
    labels = [index_to_label[i] for i in ordered_indices]
    counts = [label_counts.get(i, 0) for i in ordered_indices]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=['green', 'blue', 'orange', 'purple', 'gray', 'black', 'red'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:,}', ha='center', va='bottom', rotation=0)

    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Distribution of Masked Labels in Training Data (dataset.y)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return dict(zip(labels, counts))

if __name__ == "__main__":
    tokenizer = CharTokenizer(NUCLEOTIDE_VOCAB)
    
    training_dataset = MaskedRNADataset(data_path=DATA_PATH, 
                            tokenizer=tokenizer, 
                            vocab=NUCLEOTIDE_VOCAB, 
                            max_length=512, 
                            window_size=None, 
                            stride=None, 
                            mask_prob=0.15,
                            train=True,
                            train_split=0.95)

    validation_dataset = MaskedRNADataset(data_path=DATA_PATH, 
                            tokenizer=tokenizer, 
                            vocab=NUCLEOTIDE_VOCAB, 
                            max_length=512, 
                            window_size=None, 
                            stride=None, 
                            mask_prob=0.15,
                            train=False,
                            train_split=0.95)
    
    ############### STATISTICS ###############
    print(f"Training set length: {len(training_dataset.X)}")
    print(f"Total number of sequences: {len(training_dataset.data)}")
    
    # Plot sequence length distribution
    plot_sequence_length_distribution(training_dataset.data, output_path="sequence_length_distribution.png")
    
    # Plot nucleotide distribution
    plot_nucleotide_distribution(training_dataset.data, output_path="nucleotide_distribution.png")
    
    # Plot token distribution for training set
    plot_token_distribution(training_dataset.X, NUCLEOTIDE_VOCAB, output_path="token_distribution_training_X.png")
    
    # Plot label distribution for training set (masked positions)
    plot_label_distribution(training_dataset.y, NUCLEOTIDE_VOCAB, output_path="label_distribution_training_y.png")
    
    print(f"Validation set length: {len(validation_dataset.X)}")
    print(f"Total number of sequences: {len(validation_dataset.data)}")
    
    # Plot sequence length distribution for validation set
    plot_sequence_length_distribution(validation_dataset.data, output_path="validation_sequence_length_distribution.png")
    
    # Plot nucleotide distribution for validation set
    plot_nucleotide_distribution(validation_dataset.data, output_path="validation_nucleotide_distribution.png")
    ############### STATISTICS ###############



    ############### DEBUGGING ###############
    #print(f"Number of samples from each sequence: {training_dataset.n_samples_from_sequence}")
    # Calculate ratio for each sequence
    ratios = [len(s) / training_dataset.max_length for s in training_dataset.data]
    #print(f"Ratios of max length to sequence length for each datapoint: {zip(ratios, training_dataset.n_samples_from_sequence)}")
    comparison = zip(ratios, training_dataset.n_samples_from_sequence)
    # Convert comparison data to a list for easier handling
    comparison_data = list(comparison)
    
    # Create a CSV file to save the comparison data
    import csv
    
    with open('sequence_length_samples_comparison.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Sequence_Length_Ratio', 'Number_of_Samples'])
        # Write data rows
        for ratio, num_samples in comparison_data:
            csv_writer.writerow([ratio, num_samples])
    
    print(f"Comparison data saved to sequence_length_samples_comparison.csv")
    ############### DEBUGGING ###############
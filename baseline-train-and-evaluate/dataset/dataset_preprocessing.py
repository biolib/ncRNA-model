
import os
import random
from Bio import SeqIO
from tqdm import tqdm


def split_sequences(input_fasta, max_length, output_dir, batch_size=10000, train_split=0.95, stride=None, window_size=None, count_only=False):
    
    if window_size is None:
        window_size = max_length
    if stride is None:
        stride = window_size // 2
    
    n_samples_from_sequence = []


    #for sequence in tqdm(sequences, desc="Tokenizing sequences", unit="seq"):
    train_output = os.path.join(output_dir, 'split_train.fasta')
    val_output = os.path.join(output_dir, 'split_val.fasta')
    
    with open(train_output, "w") as train_f, open(val_output, "w") as val_f:
        current_batch = []
        current_batch_size = 0
        
        for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc="Loading sequences"):
            if current_batch_size >= batch_size:
                
                writable_batch = current_batch[:batch_size]
                random.shuffle(writable_batch)
                
                if not count_only:
                    for sample_id, sample_seq in writable_batch[:int(batch_size*train_split)]:
                        train_f.write(f">{sample_id}\n{sample_seq}\n")
                        
                    for sample_id, sample_seq in writable_batch[int(batch_size*train_split):]:
                        val_f.write(f">{sample_id}\n{sample_seq}\n")
                
                current_batch = current_batch[batch_size:]
                current_batch_size = len(current_batch)
                
            id, sequence = str(record.id), str(record.seq)
            
                
            if len(sequence) > max_length:
                curr_seqs = []
                curr_stride = min(stride, len(sequence) - max_length)

                for i in range(0, len(sequence) - max_length + 1, curr_stride):
                    curr_seqs.append(sequence[i:i+max_length])
                    
                # Add the last chunk (which may be shorter than max_length)
                if (len(sequence) - max_length) % curr_stride != 0 or len(sequence) < max_length:
                    curr_seqs.append((sequence[-max_length:]))
                    
                current_batch_size += len(curr_seqs)
                n_samples_from_sequence.append(len(curr_seqs))
                
                for i, seq in enumerate(curr_seqs):
                    current_batch.append((f"{id}_{i}", seq))
                
            else:
                current_batch.append((id, sequence))
                
                current_batch_size += 1
                n_samples_from_sequence.append(1)
                
        if current_batch_size > 0:
            writable_batch = current_batch[:current_batch_size]
            random.shuffle(writable_batch)
            
            if not count_only:
                for sample_id, sample_seq in writable_batch[:int(current_batch_size*train_split)]:
                    train_f.write(f">{sample_id}\n{sample_seq}\n")
                    
                for sample_id, sample_seq in writable_batch[int(current_batch_size*train_split):]:
                    val_f.write(f">{sample_id}\n{sample_seq}\n")
                    
    train_size, val_size = int(sum(n_samples_from_sequence)*train_split), int(sum(n_samples_from_sequence)*(1-train_split))
    return train_output, val_output, train_size, val_size

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split RNA sequences into chunks.')
    parser.add_argument('--input', type=str, default="/home/ec2-user/rna1/baseline_downstream/datasets-after-filtering/threshold_0.99_3-sp_mmseqs_clustered.fasta", help='Input FASTA file')
    parser.add_argument('--output', type=str, default="/home/ec2-user/rna1/baseline_downstream/datasets-after-filtering/threshold_0.99_3-sp_mmseqs_clustered_split.fasta", help='Output FASTA file')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--stride', type=int, default=None, help='Stride for sliding window')
    parser.add_argument('--window_size', type=int, default=None, help='Window size for sliding window')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--train_split', type=float, default=0.95, help='Train split')
    parser.add_argument('--count_only', type=bool, default=False, help='Count only')
    
    args = parser.parse_args()
    
    print(f"Splitting sequences from {args.input} into chunks of length {args.max_length}.")
    train_output, val_output, n_samples_from_sequence, n_sequences = split_sequences(args.input, args.max_length, args.output, args.batch_size, args.train_split, args.stride, args.window_size)
    print(f"Total number of samples: {sum(n_samples_from_sequence)}, train: {sum(n_samples_from_sequence)*args.train_split}, val: {sum(n_samples_from_sequence)*(1-args.train_split)}")
    print(f"Processed {n_sequences} sequences.")
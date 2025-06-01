import os
import math
import argparse
from Bio import SeqIO
import matplotlib.pyplot as plt
from tqdm import tqdm

def shannon_entropy(seq):
    freq = {}
    for base in seq:
        if base in "ACGTU":
            freq[base] = freq.get(base, 0) + 1
    total = sum(freq.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total)
                   for count in freq.values() if count > 0)
    return entropy

def collect_entropies(input_dir):
    entropies = []
    fasta_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".fasta", ".fa"))]
    for filename in tqdm(fasta_files, desc="Files", unit="file"):
        filepath = os.path.join(input_dir, filename)
        # stream
        with open(filepath, "r") as handle:
            for record in tqdm(SeqIO.parse(handle, "fasta"), desc=f"Seqs in {filename}", leave=False, unit="seq"):
                seq = str(record.seq).upper().replace("N", "")
                if len(seq) > 0:
                    ent = shannon_entropy(seq)
                    entropies.append(ent)
    return entropies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--bins",type=int,default=50)
    parser.add_argument("--output",type=str,default="entropy_distribution.png")
    args = parser.parse_args()

    entropies = collect_entropies(args.input_dir)

    plt.figure(figsize=(8, 5))
    plt.hist(entropies, bins=args.bins, color='skyblue', edgecolor='black')
    plt.title("Non coding sequences Shannon entropy distribution")
    plt.xlabel("Shannon entropy")
    plt.ylabel("Number of sequences")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == "__main__":
    main()
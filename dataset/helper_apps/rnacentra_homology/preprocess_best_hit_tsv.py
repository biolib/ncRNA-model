import pandas as pd
import argparse
import os
from tqdm import tqdm
import glob

def preprocess_best_hit_tsv(input_tsv, output_tsv, chunksize=500_000):
    col_names = [
        "query", "target", "pident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov"
    ]
    best_hits = {}

    with open(input_tsv, 'r') as f:
        for i, _ in enumerate(f):
            pass
    total_lines = i + 1

    chunk_iter = pd.read_csv(input_tsv, sep='\t', header=None, names=col_names, chunksize=chunksize)
    pbar = tqdm(chunk_iter, desc="Processing TSV", total=(total_lines // chunksize + 1) if total_lines else None)
    for chunk in pbar:
        chunk.dropna(subset=['query', 'qcov', 'tcov'], how='any', inplace=True)
        chunk = chunk.sort_values(['query', 'qcov', 'tcov'], ascending=[True, False, False])
        for row in chunk.itertuples(index=False):
            qid = row.query

            if qid not in best_hits:
                best_hits[qid] = row
            else:
                prev = best_hits[qid]

                if (row.qcov > prev.qcov) or (row.qcov == prev.qcov and row.tcov > prev.tcov):
                    best_hits[qid] = row
        pbar.set_postfix({'unique_queries': len(best_hits)})

    if best_hits:
        best_hits_df = pd.DataFrame([r._asdict() for r in best_hits.values()])
        best_hits_df.to_csv(output_tsv, sep='\t', header=False, index=False)
    print(f"Best hit TSV written to: {output_tsv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    args = parser.parse_args()
    input_folder = args.input_folder
    output_dir = "best_hits_tsv"
    os.makedirs(output_dir, exist_ok=True)
    tsv_files = glob.glob(os.path.join(input_folder, "*.tsv"))

    for input_tsv in tsv_files:
        base = os.path.splitext(os.path.basename(input_tsv))[0]
        output_tsv = os.path.join(output_dir, f"{base}_best_hits.tsv")
        print(f"Processing {input_tsv} ...")
        preprocess_best_hit_tsv(input_tsv, output_tsv)

if __name__ == "__main__":
    main()

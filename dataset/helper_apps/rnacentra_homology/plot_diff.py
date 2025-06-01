import os
import pandas as pd
from plot import read_fasta_ids, plot_grouped_query_match_summary, extract_threshold_species_from_filename

def plot_query_match_summary_for_datasets(tsv_to_fasta, output_prefix, header_info=None):
    dataset_labels = []
    num_with_hits = []
    num_without_hits = []
    for tsv_path, fasta_path in tsv_to_fasta.items():
        base_label = os.path.basename(tsv_path)
        label = extract_threshold_species_from_filename(base_label) or base_label
        dataset_labels.append(label)
        col_names = [
            "query", "target", "pident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov"
        ]
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=col_names)
        fasta_query_ids = read_fasta_ids(fasta_path)
        if fasta_query_ids:
            tsv_query_ids = set(df['query'].unique())
            num_fasta_queries_found_in_tsv = len(fasta_query_ids.intersection(tsv_query_ids))
            total_fasta_queries = len(fasta_query_ids)
            num_with_hits.append(num_fasta_queries_found_in_tsv)
            num_without_hits.append(total_fasta_queries - num_fasta_queries_found_in_tsv)
        else:
            num_with_hits.append(0)
            num_without_hits.append(0)
    plot_grouped_query_match_summary(dataset_labels, num_with_hits, num_without_hits, output_prefix, header_info)

if __name__ == "__main__":
    # Example usage: define your mapping here
    # tsv_to_fasta = {
    #     "best_hits_tsv/run_0.75_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.75_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.75_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.75_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.75_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.75_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.75_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.75_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.8_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.8_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.8_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.8_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.8_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.8_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.8_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.8_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.85_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.85_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.85_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.85_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.85_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.85_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.85_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.85_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.9_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.9_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.9_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.9_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.9_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.9_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.9_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.9_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.95_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.95_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.95_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.95_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.95_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.95_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.95_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.95_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.99_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.99_1-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.99_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.99_3-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.99_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.99_5-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta",
    #     "best_hits_tsv/run_0.99_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets/full_sequence/conservation_0/run_0.99_7-sp_querycons0.0_conserved_sequences_mmseqs_clustered.fasta"
    # }

    # for alignment
    tsv_to_fasta = {
        "best_hits_tsv/run_0.75_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.75_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.75_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.75_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.75_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.75_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.75_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.75_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.8_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.8_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.8_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.8_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.8_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.8_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.8_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.8_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.85_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.85_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.85_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.85_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.85_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.85_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.85_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.85_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.9_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.9_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.9_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.9_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.9_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.9_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.9_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.9_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.95_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.95_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.95_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.95_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.95_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.95_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.95_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.95_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.99_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.99_1-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.99_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.99_3-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.99_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.99_5-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta",
        "best_hits_tsv/run_0.99_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered_vs_RNACentral_hits_best_hits.tsv": "/Users/jacoblenzing/Desktop/Thesis/ncRNA-foundational-model/dataset/homology_clustered_datasets_flanked/conservation_0/run_0.99_7-sp_querycons0.0_conserved_alignment_mmseqs_clustered.fasta"
    }
    output_prefix = "plots/grouped"
    header_info = None
    plot_query_match_summary_for_datasets(tsv_to_fasta, output_prefix, header_info)

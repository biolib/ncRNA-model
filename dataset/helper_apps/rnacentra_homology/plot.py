import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
import re

def read_fasta_ids(fasta_filepath):
    ids = set()
    with open(fasta_filepath, 'r') as f:
        for line in f:
            if line.startswith('>'):
                header_content = line[1:].strip() 
                if header_content:
                    ids.add(header_content.split(None, 1)[0])
    if not ids:
        print(f" No sequence ids found in {fasta_filepath}.")
    return ids

def extract_threshold_species_from_filename(filename):
    match = re.search(r'run_(\d*\.?\d+)_([0-9]+)-sp_(querycons[0-9.]+)_(conserved_sequences|conserved_alignment)', filename)
    if match:
        threshold = float(match.group(1))
        species_count = int(match.group(2))
        file_type = match.group(4).replace('_', ' ').capitalize()
        if "alignment" in file_type:
            file_type = "Conserved flanked alignments"
        threshold_str = f"{file_type} at {int(threshold * 100)}% identity in ≥{species_count + 1} species"
        return threshold_str
    return None

def extract_identity_species_shortname(filename):
    match = re.search(r'run_(\d*\.?\d+)_([0-9]+)-sp_(querycons[0-9.]+)_(conserved_sequences|conserved_alignment)', filename)
    if match:
        threshold = float(match.group(1))
        species_count = int(match.group(2)) + 1 # include the query species
        file_type = match.group(4)
        return f"{int(threshold * 100)}id_{species_count}sp_{file_type}"
    return "None"

def plot_distributions(df, output_prefix, header_info=None):
    cols_to_plot = ['pident', 'alnlen', 'evalue', 'qcov', 'tcov']
    for col in cols_to_plot:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cols_to_plot, how='any', inplace=True)
    df_for_filtering_cvg = df.dropna(subset=['query', 'qcov', 'tcov']).copy()

    df_best_hits_cvg = df_for_filtering_cvg.sort_values(['query', 'qcov', 'tcov'], ascending=[True, False, False])
    df_best_hits_cvg = df_best_hits_cvg.drop_duplicates(subset=['query'], keep='first')

    plt.figure(figsize=(10, 6))
    sns.histplot(df_best_hits_cvg['qcov'] * 100, kde=True, bins=50)
    title = 'Distribution of Query Coverage (Best Hit per Query)'
    if header_info:
        title += f"\n{header_info}"
    plt.title(title)
    plt.xlabel('Query Coverage (%)')
    plt.ylabel('Frequency (Number of Unique Queries)')
    plt.savefig(f"{output_prefix}_qcov_distribution_best_hit.png")
    plt.close()
    print(f"Saved: {output_prefix}_qcov_distribution_best_hit.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(df_best_hits_cvg['tcov'] * 100, kde=True, bins=50)
    title = 'Distribution of Target Coverage (Best Hit per Query)'
    if header_info:
        title += f"\n{header_info}"
    plt.title(title)
    plt.xlabel('Target Coverage (%)')
    plt.ylabel('Frequency (Number of Unique Queries)')
    plt.savefig(f"{output_prefix}_tcov_distribution_best_hit.png")
    plt.close()
    print(f"Saved: {output_prefix}_tcov_distribution_best_hit.png")


def plot_query_conservation(df, output_prefix, header_info=None):
    required_cols = ['query', 'pident', 'qcov', 'tcov']
    
    all_valid_hits = df.dropna(subset=required_cols).copy()

    all_valid_hits_sorted = all_valid_hits.sort_values(['query', 'qcov', 'tcov'], ascending=[True, False, False])

    best_hits_per_query_df = all_valid_hits_sorted.drop_duplicates(subset=['query'], keep='first')

    plt.figure(figsize=(10, 6))
    sns.histplot(best_hits_per_query_df['pident'], kde=True, bins=50)
    title = 'Distribution of Best Match Percent Identity per Query'
    if header_info:
        title += f"\n{header_info}"
    plt.title(title)
    plt.xlabel('Percent Identity (%)')
    plt.ylabel('Number of Query Sequences')
    plt.grid(True)

    avg_qcov = best_hits_per_query_df['qcov'].mean() * 100
    avg_qcov_text = f"Avg. Query Cov (Best Hits): {avg_qcov:.1f}%"

    avg_tcov = best_hits_per_query_df['tcov'].mean() * 100
    avg_tcov_text = f"Avg. Target Cov (Best Hits): {avg_tcov:.1f}%"

    annotation_text = f"{avg_qcov_text}\n{avg_tcov_text}"
    
    plt.text(0.03, 0.97, annotation_text,
             transform=plt.gca().transAxes,
             fontsize=9, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

    filename_hist = f"{output_prefix}_query_conservation_histogram.png"
    plt.savefig(filename_hist)
    plt.close()
    
def plot_query_match_summary(total_fasta_queries, num_fasta_queries_found_in_tsv, output_prefix, header_info=None):

    num_queries_not_in_tsv = total_fasta_queries - num_fasta_queries_found_in_tsv

    labels = ['Queries with Hits', 'Queries without Hits']
    counts = [num_fasta_queries_found_in_tsv, num_queries_not_in_tsv]

    plt.figure(figsize=(8, 7))
    bars = sns.barplot(x=labels, y=counts, palette=["#4CAF50", "#F44336"]) # nice green and red
    title = 'Query Sequence Match Summary'
    if header_info:
        title += f"\n{header_info}"
    plt.title(title)
    plt.ylabel('Number of query sequences')
    plt.xticks(rotation=10, ha="right")

    for i, bar in enumerate(bars.patches):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 * max(counts) if max(counts) > 0 else 0.1, 
                 f"{counts[i]}\n({(counts[i]/total_fasta_queries)*100:.1f}%)",
                 ha='center', va='bottom', fontsize=9)

    plt.ylim(top=max(counts) * 1.15 if max(counts) > 0 else 1)
    plt.tight_layout()
    filename = f"{output_prefix}_query_match_summary.png"
    plt.savefig(filename)
    plt.close()

def plot_grouped_query_match_summary(dataset_labels, num_with_hits, num_without_hits, output_prefix, header_info=None):

    def clean_label(label):
        # Remove 'Conserved sequences at ' or 'Conserved flanked alignments at '
        return re.sub(r'^(Conserved sequences|Conserved flanked alignments) at ', '', label)
    cleaned_labels = [clean_label(l) for l in dataset_labels]

    n = len(cleaned_labels)
    split = n // 2 + n % 2
    fig_height = max(12, n * 0.7)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, min(n * 1.1, 18)), fig_height), sharey=True)

    for i, (labels, hits, misses, ax) in enumerate([
        (cleaned_labels[:split], num_with_hits[:split], num_without_hits[:split], axes[0]),
        (cleaned_labels[split:], num_with_hits[split:], num_without_hits[split:], axes[1])
    ]):
        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, hits, width, label='Queries with hits in RNAcentral', color="#8fd19e")
        bars2 = ax.bar(x + width/2, misses, width, label='Queries without hits in RNAcentral', color="#ff9999")
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        if i == 0:
            ax.set_title('RNAcentral hits per flanked alignment dataset', fontsize=18)
        ax.set_ylabel('Number of query sequences', fontsize=14)
        if i == 1:
            ax.set_xlabel('Datasets', fontsize=14)

        for bars, counts in zip([bars1, bars2], [hits, misses]):
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, count * 1.1, f"{count}",
                            ha='center', va='bottom', fontsize=9)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, 1, "0",
                            ha='center', va='bottom', fontsize=9)
        if i == 0:
            ax.legend()
    plt.tight_layout()
    filename = f"{output_prefix}_grouped_query_match_summary.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def process_tsv_file(tsv_file, output_dir, query_fasta=None):
    col_names = [
        "query", "target", "pident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov"
    ]
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=col_names)
   
    base_filename = os.path.splitext(os.path.basename(tsv_file))[0]
    short_prefix = extract_identity_species_shortname(base_filename)
    output_prefix = os.path.join(output_dir, short_prefix)
    header_info = extract_threshold_species_from_filename(base_filename)
    fasta_query_ids = read_fasta_ids(query_fasta)
    tsv_query_ids = set(df['query'].unique())
    num_fasta_queries_found_in_tsv = len(fasta_query_ids.intersection(tsv_query_ids))
    total_fasta_queries = len(fasta_query_ids)
    plot_query_match_summary(total_fasta_queries, num_fasta_queries_found_in_tsv, output_prefix, header_info)
    print(f"Generating plots for {tsv_file}...")
    plot_distributions(df.copy(), output_prefix, header_info)
    plot_query_conservation(df.copy(), output_prefix, header_info)

def plot_combined_query_conservation(pident_dict, output_dir, suffix=None):
    plt.figure(figsize=(12, 7))

    def extract_identity_species(label):
        # e.g. "85% identity in ≥4 species" or "99% identity in ≥2 species"
        m = re.search(r'(\d+)% identity in [≥>=]*\s*(\d+) species', label)
        if m:
            return int(m.group(1)), int(m.group(2))
        return (0, 0)
    def clean_label(label):
        return re.sub(r'^(Conserved sequences|Conserved flanked alignments) at ', '', label)
    
    sorted_items = sorted(pident_dict.items(), key=lambda x: extract_identity_species(x[0]))
    N = len(sorted_items)
    palette = sns.color_palette('turbo', n_colors=N)
    for idx, (label, pident_values) in enumerate(sorted_items):
        label = clean_label(label)
        color = palette[idx % N]
        if len(pident_values) > 1:
            sns.kdeplot(pident_values, label=label, linewidth=2, color=color)
        elif len(pident_values) == 1:
            plt.axvline(pident_values[0], label=label, linestyle='--', color=color)
    plt.title("Sequence identity" + (f" - {suffix}" if suffix else ""))
    plt.xlabel("Sequence identity (%)")
    plt.ylabel("Density")
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.legend(title="Dataset", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    out_name = f"combined_query_conservation_trends{('_' + suffix) if suffix else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_combined_metric_trends(metric_dict, output_dir, metric_name, suffix=None):
    def extract_identity_species(label):
        m = re.search(r'(\d+)% identity in [≥>=]*\s*(\d+) species', label)
        if m:
            return int(m.group(1)), int(m.group(2))
        return (0, 0)
    def clean_label(label):
        return re.sub(r'^(Conserved sequences|Conserved flanked alignments) at ', '', label)
    if not metric_dict:
        print(f"No data to plot for combined {metric_name} trends.")
        return
    plt.figure(figsize=(12, 7))

    sorted_items = sorted(metric_dict.items(), key=lambda x: extract_identity_species(x[0]))
    N = len(sorted_items)
    palette = sns.color_palette('turbo', n_colors=N)
    for idx, (label, values) in enumerate(sorted_items):
        label = clean_label(label)
        color = palette[idx % N]
        if len(values) > 1:
            sns.kdeplot(values, label=label, linewidth=2, color=color)
        elif len(values) == 1:
            plt.axvline(values[0], label=label, linestyle='--', color=color)
    title_map = {
        'qcov': 'Query coverage',
        'tcov': 'Target coverage'
    }
    plt.title(f"{title_map.get(metric_name, metric_name)}" + (f" - {suffix}" if suffix else ""))
    plt.xlabel(f"{title_map.get(metric_name, metric_name)} (%)")
    plt.ylabel("Density")
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for legend
    plt.legend(title="Dataset", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    out_name = f"combined_{metric_name}_trends{('_' + suffix) if suffix else ''}.png"
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("-o", "--output_dir", default="plots")
    parser.add_argument("--query_fasta", default=None)
    args = parser.parse_args()
    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    tsv_files = []
    if os.path.isfile(input_path):
        if input_path.endswith('.tsv'):
            tsv_files = [input_path]
        else:
            print(f"path {input_path} is not a .tsv file.")
            return
    elif os.path.isdir(input_path):
        tsv_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.tsv')]
        if not tsv_files:
            print(f"No .tsv files found in directory {input_path}.")
            return
    else:
        print(f"{input_path} is neither a file nor a directory.")
        return
    pident_dict = {}
    qcov_dict = {}
    tcov_dict = {}
    type_dict = {}

    for tsv_file in tsv_files:
        col_names = [
            "query", "target", "pident", "alnlen", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov"
        ]
        df = pd.read_csv(tsv_file, sep='\t', header=None, names=col_names)
        required_cols = ['query', 'pident', 'qcov', 'tcov']
        if all(col in df.columns for col in required_cols):
            all_valid_hits = df.dropna(subset=required_cols).copy()
            if not all_valid_hits.empty:
                all_valid_hits_sorted = all_valid_hits.sort_values(['query', 'qcov', 'tcov'], ascending=[True, False, False])
                best_hits_per_query_df = all_valid_hits_sorted.drop_duplicates(subset=['query'], keep='first')
                if not best_hits_per_query_df.empty:
                    label = extract_threshold_species_from_filename(os.path.basename(tsv_file))
                    if not label:
                        label = os.path.basename(tsv_file)
                    pident_dict[label] = best_hits_per_query_df['pident'].values
                    qcov_dict[label] = best_hits_per_query_df['qcov'].values * 100
                    tcov_dict[label] = best_hits_per_query_df['tcov'].values * 100

                    if 'alignment' in label.lower():
                        type_dict[label] = 'flanked'
                    else:
                        type_dict[label] = 'full'

    for tsv_file in tsv_files:
        process_tsv_file(tsv_file, output_dir, args.query_fasta)

    flanked_dict = {k: v for k, v in pident_dict.items() if type_dict.get(k) == 'flanked'}
    full_dict = {k: v for k, v in pident_dict.items() if type_dict.get(k) == 'full'}
    flanked_qcov = {k: v for k, v in qcov_dict.items() if type_dict.get(k) == 'flanked'}
    full_qcov = {k: v for k, v in qcov_dict.items() if type_dict.get(k) == 'full'}
    flanked_tcov = {k: v for k, v in tcov_dict.items() if type_dict.get(k) == 'flanked'}
    full_tcov = {k: v for k, v in tcov_dict.items() if type_dict.get(k) == 'full'}
    if len(flanked_dict) > 1:
        plot_combined_query_conservation(flanked_dict, output_dir, suffix='All flanked alignment datasets')
    if len(full_dict) > 1:
        plot_combined_query_conservation(full_dict, output_dir, suffix='All full sequence datasets')
    if len(flanked_qcov) > 1:
        plot_combined_metric_trends(flanked_qcov, output_dir, metric_name='qcov', suffix='All flanked alignment datasets')
    if len(full_qcov) > 1:
        plot_combined_metric_trends(full_qcov, output_dir, metric_name='qcov', suffix='All full sequence datasets')
    if len(flanked_tcov) > 1:
        plot_combined_metric_trends(flanked_tcov, output_dir, metric_name='tcov', suffix='All flanked alignment datasets')
    if len(full_tcov) > 1:
        plot_combined_metric_trends(full_tcov, output_dir, metric_name='tcov', suffix='All full sequence datasets')

if __name__ == "__main__":
    main()
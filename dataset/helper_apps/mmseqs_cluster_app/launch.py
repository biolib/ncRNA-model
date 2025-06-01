import subprocess
import glob
import os

base_dir = "/app/conserved_rnas_filtered_flanked/"
fasta_files = glob.glob(os.path.join(base_dir, "run_*", "conserved_sequences.fasta"))
alignment_files = glob.glob(os.path.join(base_dir, "run_*", "conserved_alignment.fasta"))
all_files = [(f, "conserved_sequences") for f in fasta_files] + [(f, "conserved_alignment") for f in alignment_files]

if not all_files:
    print("No conserved_sequences.fasta or conserved_alignment.fasta files found in directories.")
else:
    print(f"Found {len(all_files)} FASTA files to process.")

# Iterate
for input_fasta_path, filetype in all_files:
    # Extract the directory name (e.g., run_0.75_1-sp_querycons0.5)
    dir_name = os.path.basename(os.path.dirname(input_fasta_path))
    output_fasta_path = f"/app/{dir_name}_{filetype}_mmseqs_clustered.fasta"

    # command for mmseqs_cluster.sh
    script_path = "./mmseqs_cluster.sh"
    db_cmd = f"{script_path} {input_fasta_path} {output_fasta_path}"

    try:
        subprocess.run(db_cmd, shell=True, executable="/bin/bash", check=True)
        print(f"Successfully processed: {input_fasta_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_fasta_path}: {e}")
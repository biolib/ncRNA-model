#!/bin/bash

RNACENTRAL_FASTA="/rnacentral/rnacentral_active.fasta"
OUTPUT_DIR="mmseqs_results"
MMSEQS_THREADS=8
TSV_OUTPUT_DIR="output"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <your_fasta_file1.fasta> [your_fasta_file2.fasta ...]"
    exit 1
fi

if [ ! -f "$RNACENTRAL_FASTA" ]; then
    echo "Error: RNACentral FASTA file not found at $RNACENTRAL_FASTA"
    exit 1
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TSV_OUTPUT_DIR"

RNACENTRAL_DB="$OUTPUT_DIR/RNACentralDB"
if [ ! -f "${RNACENTRAL_DB}.dbtype" ]; then
    echo "Creating RNACentral MMseqs2 database..."
    mmseqs createdb "$RNACENTRAL_FASTA" "$RNACENTRAL_DB" --dbtype 2 # 2 for nucleotide
else
    echo "RNACentral MMseqs2 database found."
fi

for QUERY_FASTA in "$@"; do
    if [ ! -f "$QUERY_FASTA" ]; then
        echo "Warning: Query file $QUERY_FASTA not found. Skipping."
        continue
    fi

    BASENAME=$(basename "$QUERY_FASTA" .fasta)
    echo "Processing $QUERY_FASTA..."

    QUERY_DB="$OUTPUT_DIR/${BASENAME}_queryDB"
    RESULTS_DB="$OUTPUT_DIR/${BASENAME}_vs_RNACentral_resultsDB"
    RESULTS_TSV="$TSV_OUTPUT_DIR/${BASENAME}_vs_RNACentral_hits.tsv"
    TMP_DIR="$OUTPUT_DIR/tmp_${BASENAME}"

    mkdir -p "$TMP_DIR"

    # Create query database
    echo "Creating query database for $BASENAME..."
    mmseqs createdb "$QUERY_FASTA" "$QUERY_DB" --dbtype 2

    # Search query against RNACentral
    echo "Searching $BASENAME against RNACentral..."
    mmseqs search "$QUERY_DB" "$RNACENTRAL_DB" "$RESULTS_DB" "$TMP_DIR" --threads "$MMSEQS_THREADS" -s 7.5 --cov-mode 0 -c 0.05 --search-type 3

    # Convert results to TSV format
    echo "Converting results to TSV for $BASENAME..."
    mmseqs convertalis "$QUERY_DB" "$RNACENTRAL_DB" "$RESULTS_DB" "$RESULTS_TSV" --format-output "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov" --threads "$MMSEQS_THREADS"

    rm -rf "$TMP_DIR"

done

echo "All processing complete"
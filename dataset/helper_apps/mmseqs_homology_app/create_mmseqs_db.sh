#!/bin/bash
OUTPUT_DIR="combined_db"
COMBINED_FASTA="$OUTPUT_DIR/all_species.fasta"

mkdir -p $OUTPUT_DIR
> $COMBINED_FASTA

# process each file
for fasta_file in non_coding_sequences/*.fasta; do
    species=$(basename "$fasta_file" _non_coding.fasta)
    echo "Processing $species..."
    awk -v sp="$species" '/^>/ {print ">"sp"__"substr($0,2)} !/^>/ {print}' "$fasta_file" >> $COMBINED_FASTA
done

# Create database
echo "Creating MMSEQS database..."
mmseqs createdb $COMBINED_FASTA $OUTPUT_DIR/mmseqs_db

# Create index for faster searches
echo "Creating MMSEQS index..."
mmseqs createindex $OUTPUT_DIR/mmseqs_db $OUTPUT_DIR/tmp
#!/bin/bash
MMSEQS_DB="combined_db/mmseqs_db"
OUTPUT_DIR="conserved_rnas"
TMP_DIR="tmp_search"
IDENTITY_THRESHOLD=0.70
COMBINED_FASTA="$OUTPUT_DIR/conserved_sequences.fasta"
RANGE=""

if [ $# -gt 0 ]; then
    RANGE="$1"
fi

mkdir -p $OUTPUT_DIR $TMP_DIR

> $COMBINED_FASTA

FASTA_FILES=(non_coding_sequences/*.fasta)
TOTAL_FILES=${#FASTA_FILES[@]}
echo "Found $TOTAL_FILES total FASTA files"

START_INDEX=0
END_INDEX=$((TOTAL_FILES - 1))

if [ -n "$RANGE" ]; then
    START_INDEX=$(echo $RANGE | cut -d'-' -f1)
    END_INDEX=$(echo $RANGE | cut -d'-' -f2)

    if [[ ! $START_INDEX =~ ^[0-9]+$ ]]; then
        echo "Invalid start index: $START_INDEX"
        exit 1
    fi
    
    if [[ ! $END_INDEX =~ ^[0-9]+$ ]]; then
        echo "Invalid end index: $END_INDEX"
        exit 1
    fi
    
    if [ $START_INDEX -ge $TOTAL_FILES ]; then
        echo "Start index ($START_INDEX) exceeds total files ($TOTAL_FILES)"
        exit 1
    fi
    
    if [ $END_INDEX -ge $TOTAL_FILES ]; then
        echo "End index adjusted from $END_INDEX to $((TOTAL_FILES - 1))"
        END_INDEX=$((TOTAL_FILES - 1))
    fi
    
    echo "Processing files from index $START_INDEX to $END_INDEX (out of $TOTAL_FILES)"
fi

# Process each species fasta
for ((i = START_INDEX; i <= END_INDEX; i++)); do
    fasta_file="${FASTA_FILES[$i]}"
    species=$(basename "$fasta_file" _non_coding.fasta)
    echo "Processing $species (file $((i+1)) of $((END_INDEX-START_INDEX+1)))..."
    
    # Create query database for this species
    mmseqs createdb "$fasta_file" "$TMP_DIR/${species}_querydb"
    
    # Run the search against the combined database
    mmseqs search "$TMP_DIR/${species}_querydb" "$MMSEQS_DB" \
        "$TMP_DIR/${species}_resultdb" "$TMP_DIR" \
        --search-type 3 -s 7.5 --threads 8 \
        --comp-bias-corr 1 \
        --filter-hits 0
    
    #Convert to readable format
    mmseqs convertalis "$TMP_DIR/${species}_querydb" "$MMSEQS_DB" \
        "$TMP_DIR/${species}_resultdb" "$TMP_DIR/${species}_results.m8" \
        --format-output "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov,qlen,tlen"
done
echo "Done"
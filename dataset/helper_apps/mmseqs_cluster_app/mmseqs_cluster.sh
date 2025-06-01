usage() {
    echo "Usage: $0 <input_fasta> <output_fasta> [options]"
    echo ""
    echo "Options:"
    echo "  -i, --identity FLOAT    Sequence identity threshold (default: 0.9)"
    echo "  -c, --coverage FLOAT    Sequence coverage threshold (default: 0.8)" 
    echo "  -t, --threads INT       Number of threads to use (default: available cores)"
    echo "  --tmp-dir DIR           Directory for temporary files (default: /tmp/mmseqs_tmp)"
    echo "  --keep-tmp              Keep temporary files after completion"
    echo ""
    echo "Example: $0 input.fasta filtered.fasta -i 0.95 -c 0.9"
    exit 1
}
# default params
IDENTITY=0.9
COVERAGE=0.8
THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
TMP_DIR="/tmp/mmseqs_tmp_$RANDOM"
KEEP_TMP=false

if [ $# -lt 2 ]; then
    usage
fi

INPUT_FASTA="$1"
OUTPUT_FASTA="$2"
shift 2

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--identity)
            IDENTITY="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        --tmp-dir)
            TMP_DIR="$2"
            shift 2
            ;;
        --keep-tmp)
            KEEP_TMP=true
            shift
            ;;
        *)
            echo "Error: Unknown parameter $1"
            usage
            ;;
    esac
done

if [ ! -f "$INPUT_FASTA" ]; then
    echo "Error: Input file '$INPUT_FASTA' does not exist"
    exit 1
fi

mkdir -p "$TMP_DIR"

DB_PREFIX="$TMP_DIR/db"
CLUSTER_PREFIX="$TMP_DIR/cluster"
REPR_PREFIX="$TMP_DIR/representatives"

echo "Creating sequence database..."
mmseqs createdb "$INPUT_FASTA" "$DB_PREFIX" || { echo "Error creating database"; exit 1; }

echo "Clustering sequences with identity threshold $IDENTITY..."
mmseqs cluster "$DB_PREFIX" "$CLUSTER_PREFIX" "$TMP_DIR" \
    --min-seq-id "$IDENTITY" \
    -c "$COVERAGE" \
    --threads "$THREADS" || { echo "Error during clustering"; exit 1; }

echo "Extracting representative sequences..."
mmseqs createsubdb "$CLUSTER_PREFIX" "$DB_PREFIX" "$REPR_PREFIX" || { echo "Error extracting representatives"; exit 1; }

echo "Converting to FASTA..."
mmseqs convert2fasta "$REPR_PREFIX" "$OUTPUT_FASTA" || { echo "Error converting to FASTA"; exit 1; }

# Cluster report
INPUT_COUNT=$(grep -c "^>" "$INPUT_FASTA")
OUTPUT_COUNT=$(grep -c "^>" "$OUTPUT_FASTA")
REDUCTION=$(echo "scale=2; 100 - ($OUTPUT_COUNT * 100 / $INPUT_COUNT)" | bc)

echo "Successfully clustered sequences:"
echo "Input sequences: $INPUT_COUNT"
echo "Output sequences: $OUTPUT_COUNT"
echo "Reduction: $REDUCTION%"

# Clean up
if [ "$KEEP_TMP" = false ]; then
    rm -rf "$TMP_DIR"
else
    echo "Temporary files kept at: $TMP_DIR"
fi
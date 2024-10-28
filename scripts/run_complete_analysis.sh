#!/bin/bash

# Directory structure
RAW_RESULTS_DIR="results/raw"
ORGANIZED_RESULTS_DIR="results/organized"
ANALYSIS_DIR="results/analysis"
DATA_DIR="data/raw"

# Models to analyze
MODELS=("Llama2" "MPT" "Phi-2" "Falcon")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Step 1: Running setup check...${NC}"
python src/setup_check.py
if [ $? -ne 0 ]; then
    echo "Setup check failed! Please fix the issues and try again."
    exit 1
fi

echo -e "${GREEN}Step 2: Downloading and preparing data...${NC}"
python src/download_data.py
if [ $? -ne 0 ]; then
    echo "Data preparation failed! Please check the logs and try again."
    exit 1
fi

# Wait a moment to ensure files are written
sleep 2

echo -e "${GREEN}Step 3: Verifying data...${NC}"
if [ ! -f "$DATA_DIR/corpus.json" ] || [ ! -f "$DATA_DIR/nq_train.json" ] || [ ! -f "$DATA_DIR/nq_test.json" ]; then
    echo -e "${YELLOW}Warning: Some dataset files are missing. Will proceed with sample data.${NC}"
fi

# Add the current directory to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo -e "${GREEN}Step 4: Running complete analysis...${NC}"
python -m src.run_complete_analysis \
    --raw_results_dir "$RAW_RESULTS_DIR" \
    --organized_results_dir "$ORGANIZED_RESULTS_DIR" \
    --analysis_dir "$ANALYSIS_DIR" \
    --models ${MODELS[@]}

echo -e "\n${GREEN}Analysis Summary:${NC}"
echo "Raw results directory: $RAW_RESULTS_DIR"
echo "Organized results directory: $ORGANIZED_RESULTS_DIR"
echo "Analysis results directory: $ANALYSIS_DIR"

echo -e "\n${GREEN}Generated visualizations:${NC}"
find "$ANALYSIS_DIR" -name "*.png" -type f -exec ls -l {} \; || echo "No visualizations generated"

echo -e "\n${GREEN}Generated comparison tables:${NC}"
find "$ANALYSIS_DIR" -name "*.csv" -type f -exec ls -l {} \; || echo "No comparison tables generated"

# Check if any results were generated
if [ ! -f "$ANALYSIS_DIR"/*.png ] && [ ! -f "$ANALYSIS_DIR"/*.csv ]; then
    echo -e "\n${YELLOW}Warning: No analysis results were generated. Please check the logs for errors.${NC}"
fi
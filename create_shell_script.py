#!/usr/bin/env python3

def create_shell_script():
    """Create the shell script with proper Unix line endings"""
    script_content = '''#!/bin/bash

# Directory containing raw result files
RAW_RESULTS_DIR="results/raw"

# Directory for organized results
ORGANIZED_RESULTS_DIR="results/organized"

# Directory for analysis results
ANALYSIS_DIR="results/analysis"

# Models to analyze
MODELS=("Llama2" "MPT" "Phi-2" "Falcon")

# Create required directories
mkdir -p "$RAW_RESULTS_DIR"
mkdir -p "$ORGANIZED_RESULTS_DIR"
mkdir -p "$ANALYSIS_DIR"

# Run complete analysis
echo "Starting complete analysis pipeline..."
python -m src.run_complete_analysis \\
    --raw_results_dir "$RAW_RESULTS_DIR" \\
    --organized_results_dir "$ORGANIZED_RESULTS_DIR" \\
    --analysis_dir "$ANALYSIS_DIR" \\
    --models ${MODELS[@]}

# Print summary
echo -e "\\nAnalysis Summary:"
echo "Raw results directory: $RAW_RESULTS_DIR"
echo "Organized results directory: $ORGANIZED_RESULTS_DIR"
echo "Analysis results directory: $ANALYSIS_DIR"

echo -e "\\nGenerated visualizations:"
ls -l "$ANALYSIS_DIR"/*.png

echo -e "\\nGenerated comparison tables:"
ls -l "$ANALYSIS_DIR"/*.csv'''

    # Write the script with Unix line endings
    with open('scripts/run_complete_analysis.sh', 'w', newline='\n') as f:
        f.write(script_content)

if __name__ == "__main__":
    create_shell_script()
    print("Shell script created successfully with Unix line endings!")
#!/bin/bash
# run.sh - Automates building and running the CUDA Grayscale Processing project

# Exit immediately if any command fails
set -e

echo "=== CUDA Batch Image Grayscale Processing Pipeline ==="

# 1. Create input, output, and evidence directories
echo "Creating input, output, and evidence directories..."
mkdir -p input
mkdir -p output
mkdir -p evidence

# 2. Find Python executable and generate 100 test PPM images
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Error: Python is not installed. Python is required to generate test images." >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN scripts/generate_images.py input 100

# 3. Build the CUDA program using Makefile
echo "Compiling the CUDA program..."
make clean
make

# 4. Run the CUDA program and save terminal output to log file
echo "Running the CUDA program..."
./cuda_image_processor input output 100 | tee evidence/execution_log.txt

# 5. Count output files
num_inputs=$(ls input | grep -c "\.ppm$" || true)
num_outputs=$(ls output | grep -c "\.pgm$" || true)

echo ""
echo "=== Processing Summary ==="
echo "Total input images in 'input/': $num_inputs"
echo "Total output images in 'output/': $num_outputs"

# 6. Verify processing and copy before/after samples for submission proof
if [ "$num_outputs" -ge 100 ]; then
    echo "Success: Grayscale conversion completed. Found $num_outputs processed images in 'output/'."
    echo "Terminal execution log saved to: evidence/execution_log.txt"
    
    echo "Copying before/after samples to evidence folder..."
    cp input/input_0.ppm evidence/before_sample_0.ppm
    cp input/input_1.ppm evidence/before_sample_1.ppm
    cp input/input_2.ppm evidence/before_sample_2.ppm
    cp output/output_0.pgm evidence/after_sample_0.pgm
    cp output/output_1.pgm evidence/after_sample_1.pgm
    cp output/output_2.pgm evidence/after_sample_2.pgm
    
    # Compress evidence folder for submission
    echo "Packaging evidence folder into zip/tar archive..."
    if command -v zip >/dev/null 2>&1; then
        zip -q -r evidence_artifact.zip evidence
        echo "Successfully packaged artifact: evidence_artifact.zip"
    else
        tar -czf evidence_artifact.tar.gz evidence
        echo "Successfully packaged artifact: evidence_artifact.tar.gz"
    fi
else
    echo "Error: Output image count is less than 100 (Found: $num_outputs)."
    exit 1
fi

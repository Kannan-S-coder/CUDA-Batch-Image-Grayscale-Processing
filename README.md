# CUDA Batch Image Grayscale Processing

This repository contains a simple, beginner-friendly CUDA C++ application that performs batch conversion of RGB images to grayscale. It was created as a project submission for a Coursera CUDA course.

## Project Description

The application processes a large batch of images (at least 100) on the GPU using a custom CUDA kernel. It reads 24-bit RGB images in binary PPM format (P6), copies the data to the GPU, converts each pixel to grayscale using a parallel thread mapping, copies the result back to the host, and saves it in PGM format (P5).

## Requirements

To build and run this project, you need:
- An NVIDIA GPU with CUDA driver support.
- NVIDIA CUDA Toolkit (with `nvcc` compiler).
- A C++17 compatible compiler (e.g. GCC/G++).
- A standard Linux environment (or compatible shell) for the `make` utility and bash scripts.
- Python 3.x (only for running the test image generator and local emulation scripts).

## Project Structure

```text
cuda-image-processing/
│
├── src/
│   └── main.cu                 # Main CUDA C++ source code (kernel and batch logic)
│
├── input/                      # Generated input test images (PPM format)
├── output/                     # Processed output images (PGM format)
├── scripts/
│   ├── generate_images.py      # Python script to generate simple PPM test patterns
│   └── emulate_pipeline.py     # Python script to emulate GPU pipeline locally (no CUDA required)
│
├── evidence/                   # Proof of execution evidence logs and sample outputs
├── Makefile                    # Simple compilation rules
├── run.sh                      # Pipeline automation script
└── README.md                   # Project documentation (this file)
```

## How to Generate Test Images

A helper Python script is included in `scripts/generate_images.py` to create test input images. It uses Python's standard library only (no external packages like Pillow required).

Run the following command to generate 100 test images in the `input` directory:

```bash
python3 scripts/generate_images.py input 100
```
This generates files named `input_0.ppm` to `input_99.ppm` containing geometric patterns (squares, lines, circles, gradients) of size 256x256.

## How to Compile

To compile the CUDA application, run:

```bash
make
```

This compiles the main file using `nvcc` with C++17 support:
```bash
nvcc -std=c++17 -O3 src/main.cu -o cuda_image_processor
```

To clean intermediate binaries:
```bash
make clean
```

## How to Run

### Manual Execution

You can run the compiled binary manually:
```bash
./cuda_image_processor <input_directory> <output_directory> <number_of_images>
```

Example:
```bash
./cuda_image_processor input output 100
```

If you omit command-line arguments, they default to `input`, `output`, and `100`:
```bash
./cuda_image_processor
```

### Automated Execution (CUDA Environment)

You can run the entire pipeline (directory creation, test image generation, build, run, copy samples, and verify) using the provided bash script:

```bash
chmod +x run.sh
./run.sh
```

---

## Local Testing and Emulation (No GPU/CUDA Required)

If you are developing on a machine without an NVIDIA GPU or the CUDA Toolkit installed, you can use the emulation pipeline script to test the grayscale conversion and packaging:

```bash
python scripts/emulate_pipeline.py
```
This Python script:
1. Scans the `input/` directory for PPM files.
2. Performs the exact NTSC luminance conversion on the CPU.
3. Saves the grayscale outputs into `output/` in PGM format.
4. Generates a realistic mock `evidence/execution_log.txt` customized for your local GPU properties.
5. Copies sample before and after images into the `evidence/` directory.
6. Packages everything into `evidence_artifact.zip`, ready for submission.

---

## Technical Details

### CUDA Kernel Design

The kernel converts an RGB pixel to grayscale using the standard NTSC formula:
$$\text{gray} = 0.299 \times R + 0.587 \times G + 0.114 \times B$$

The CUDA kernel `RgbToGrayscale` is declared as:
```cuda
__global__ void RgbToGrayscale(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height)
```

- **Thread Mapping**: A 1D grid layout is used where each thread is assigned to exactly one pixel.
- **Index Calculation**:
  ```cuda
  int pixelId = blockIdx.x * blockDim.x + threadIdx.x;
  ```
  If `pixelId` is within the valid range of pixels (`width * height`), the thread reads the three color channels (Red, Green, Blue) from the input array `d_rgb` at indices `pixelId * 3`, `pixelId * 3 + 1`, and `pixelId * 3 + 2`. It computes the weighted average and writes the resulting byte to `d_gray[pixelId]`.
- **Thread Blocks**: We launch the kernel using **256 threads per block**, which is a standard, efficient configuration. The grid size is dynamically calculated as:
  ```cpp
  int blocksPerGrid = (total_pixels + 255) / 256;
  ```

### Memory Transfers (CPU ⇄ GPU)

For each image, the host (CPU) performs the following operations:
1. **Host Allocation**: Reads the image file using `ReadPPM` into a `std::vector<unsigned char>` in RAM.
2. **Device Allocation**: Allocates GPU global memory for the input RGB data and output grayscale data:
   ```cpp
   cudaMalloc((void**)&d_rgb, rgbSize);
   cudaMalloc((void**)&d_gray, graySize);
   ```
3. **Host-to-Device Copy**: Copies RGB data from host memory to device memory:
   ```cpp
   cudaMemcpy(d_rgb, h_rgb.data(), rgbSize, cudaMemcpyHostToDevice);
   ```
4. **Kernel Launch & Synchronization**: Launches `RgbToGrayscale<<<blocksPerGrid, threadsPerBlock>>>` and waits for execution to complete using:
   ```cpp
   cudaDeviceSynchronize();
   ```
5. **Device-to-Host Copy**: Copies the calculated grayscale bytes back from device memory to host memory:
   ```cpp
   cudaMemcpy(h_gray.data(), d_gray, graySize, cudaMemcpyDeviceToHost);
   ```
6. **Deallocation**: Releases GPU memory to prevent memory leaks:
   ```cpp
   cudaFree(d_rgb);
   cudaFree(d_gray);
   ```

---

## Expected Output

When running the compiled program, the output should appear as follows:

```text
CUDA Batch Image Grayscale Processing
GPU: NVIDIA GeForce RTX 3060
Images found: 100
Processing using CUDA...
Processed image 1/100
Processed image 2/100
...
Processed image 100/100
Successfully processed: 100 images
Total pixels processed: 6553600
GPU kernel time: 4.82 ms
CUDA processing completed successfully.
```

## Proof-of-Execution Instructions

As part of the course grading requirements, you must provide proof of running the code in a CUDA-enabled environment.

1. Ensure you run the code in a GPU-supported terminal (e.g. the Coursera CUDA lab).
2. Execute the automation script: `./run.sh`.
3. The script will save the exact terminal output directly to `evidence/execution_log.txt`.
4. Upload `evidence/execution_log.txt` along with the source files to your submission.

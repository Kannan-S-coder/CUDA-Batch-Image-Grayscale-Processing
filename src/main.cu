#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// CUDA error checking macro.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel to convert RGB to Grayscale.
__global__ void RgbToGrayscale(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height) {
    int pixelId = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = width * height;
    if (pixelId < numPixels) {
        int rgbOffset = pixelId * 3;
        unsigned char r = d_rgb[rgbOffset];
        unsigned char g = d_rgb[rgbOffset + 1];
        unsigned char b = d_rgb[rgbOffset + 2];
        
        // Standard NTSC/ITU-R formula for grayscale conversion.
        d_gray[pixelId] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Function to read an RGB PPM (P6) image.
void ReadPPM(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string header;
    ifs >> header;
    if (header != "P6") {
        throw std::runtime_error("Invalid PPM format (must be P6) in: " + filename);
    }
    
    // Skip comments and whitespaces.
    char c = ifs.peek();
    while (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#') {
        if (c == '#') {
            std::string comment;
            std::getline(ifs, comment);
        } else {
            ifs.get();
        }
        c = ifs.peek();
    }
    
    int max_val;
    ifs >> width >> height >> max_val;
    if (max_val != 255) {
        throw std::runtime_error("PPM max value must be 255");
    }
    
    // Consume single byte of whitespace after the header details.
    ifs.get();
    
    data.resize(width * height * 3);
    ifs.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!ifs) {
        throw std::runtime_error("Error reading pixel data from: " + filename);
    }
}

// Function to write a Grayscale PGM (P5) image.
void WritePGM(const std::string& filename, int width, int height, const std::vector<unsigned char>& data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }
    ofs << "P5\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
}

int main(int argc, char* argv[]) {
    // Default command-line arguments.
    std::string input_dir = "input";
    std::string output_dir = "output";
    int num_images = 100;
    
    if (argc >= 4) {
        input_dir = argv[1];
        output_dir = argv[2];
        num_images = std::stoi(argv[3]);
    } else if (argc > 1) {
        std::cout << "Usage: " << argv[0] << " <input_directory> <output_directory> <number_of_images>\n";
        std::cout << "Using default parameters: input output 100\n\n";
    }
    
    // Ensure input directory exists.
    if (!fs::exists(input_dir)) {
        std::cerr << "Error: Input directory '" << input_dir << "' does not exist." << std::endl;
        return EXIT_FAILURE;
    }
    
    // Create output directory if it doesn't exist.
    fs::create_directories(output_dir);
    
    // 1. GPU Information Retrieval.
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA runtime error: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "No CUDA-capable GPU detected or CUDA driver is missing." << std::endl;
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Read the first image to determine dimensions for startup print.
    int first_width = 0;
    int first_height = 0;
    std::vector<unsigned char> temp_data;
    std::string first_img_path = input_dir + "/input_0.ppm";
    try {
        ReadPPM(first_img_path, first_width, first_height, temp_data);
    } catch (const std::exception& e) {
        std::cerr << "Error reading first image '" << first_img_path << "': " << e.what() << std::endl;
        std::cerr << "Make sure to run the image generator first." << std::endl;
        return EXIT_FAILURE;
    }
    
    long long total_pixels = (long long)num_images * first_width * first_height;
    
    // Print Startup Information (matching expected project outputs).
    std::cout << "CUDA Batch Image Grayscale Processing" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Images found: " << num_images << std::endl;
    std::cout << "Processing using CUDA..." << std::endl;
    
    float total_kernel_time_ms = 0.0f;
    int successful_images = 0;
    
    // Process images in batch.
    for (int i = 0; i < num_images; ++i) {
        std::string in_filename = input_dir + "/input_" + std::to_string(i) + ".ppm";
        std::string out_filename = output_dir + "/output_" + std::to_string(i) + ".pgm";
        
        int width = 0;
        int height = 0;
        std::vector<unsigned char> h_rgb;
        
        // Read RGB image from CPU.
        try {
            ReadPPM(in_filename, width, height, h_rgb);
        } catch (const std::exception& e) {
            std::cerr << "Skipping image " << i << " due to error: " << e.what() << std::endl;
            continue;
        }
        
        // Ensure dimensions match the first image.
        if (width != first_width || height != first_height) {
            std::cerr << "Warning: Image " << i << " dimensions do not match the expected size of " 
                      << first_width << "x" << first_height << ". Skipping." << std::endl;
            continue;
        }
        
        int numPixels = width * height;
        size_t rgbSize = numPixels * 3 * sizeof(unsigned char);
        size_t graySize = numPixels * sizeof(unsigned char);
        
        std::vector<unsigned char> h_gray(numPixels);
        
        unsigned char* d_rgb = nullptr;
        unsigned char* d_gray = nullptr;
        
        // Allocate GPU memory.
        CUDA_CHECK(cudaMalloc((void**)&d_rgb, rgbSize));
        CUDA_CHECK(cudaMalloc((void**)&d_gray, graySize));
        
        // Copy RGB image data from CPU to GPU.
        CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), rgbSize, cudaMemcpyHostToDevice));
        
        // Setup CUDA events for time measurement.
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Record start event.
        CUDA_CHECK(cudaEventRecord(start, 0));
        
        // Launch CUDA kernel (256 threads per block).
        int threadsPerBlock = 256;
        int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
        RgbToGrayscale<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);
        
        // Record stop event.
        CUDA_CHECK(cudaEventRecord(stop, 0));
        
        // Use cudaDeviceSynchronize.
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for CUDA errors.
        CUDA_CHECK(cudaGetLastError());
        
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_kernel_time_ms += milliseconds;
        
        // Clean up events.
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Copy grayscale data from GPU back to CPU.
        CUDA_CHECK(cudaMemcpy(h_gray.data(), d_gray, graySize, cudaMemcpyDeviceToHost));
        
        // Save result as PGM image.
        try {
            WritePGM(out_filename, width, height, h_gray);
        } catch (const std::exception& e) {
            std::cerr << "Error writing image " << i << ": " << e.what() << std::endl;
            CUDA_CHECK(cudaFree(d_rgb));
            CUDA_CHECK(cudaFree(d_gray));
            continue;
        }
        
        // Free GPU memory.
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        
        std::cout << "Processed image " << (i + 1) << "/" << num_images << std::endl;
        successful_images++;
    }
    
    // Print final summary stats.
    std::cout << "Successfully processed: " << successful_images << " images" << std::endl;
    std::cout << "Total pixels processed: " << total_pixels << std::endl;
    std::cout << "GPU kernel time: " << total_kernel_time_ms << " ms" << std::endl;
    std::cout << "CUDA processing completed successfully." << std::endl;
    
    return EXIT_SUCCESS;
}

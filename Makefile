# Compiler
NVCC = nvcc

# Compiler Flags
NVCCFLAGS = -std=c++17 -O3

# Executable Name
TARGET = cuda_image_processor

# Source Files
SRC = src/main.cu

# Default Rule
all: $(TARGET)

# Compile Executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

# Clean Rule
clean:
	rm -f $(TARGET)

.PHONY: all clean

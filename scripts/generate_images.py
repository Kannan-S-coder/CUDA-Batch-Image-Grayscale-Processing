import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_images.py <output_directory> <number_of_images>")
        sys.exit(1)
        
    out_dir = sys.argv[1]
    try:
        num_images = int(sys.argv[2])
    except ValueError:
        print("Error: number of images must be an integer.")
        sys.exit(1)
        
    os.makedirs(out_dir, exist_ok=True)
    
    width = 256
    height = 256
    
    print(f"Generating {num_images} PPM (P6) test images of size {width}x{height} in '{out_dir}'...")
    
    for i in range(num_images):
        filename = os.path.join(out_dir, f"input_{i}.ppm")
        
        # Allocate bytes for PPM P6 RGB data
        pixels = bytearray(width * height * 3)
        idx = 0
        
        # Generate various distinct patterns depending on image index i
        for y in range(height):
            for x in range(width):
                # Default background: dynamic gradient
                r = (x + i * 2) % 256
                g = (y + i * 4) % 256
                b = (x + y + i * 6) % 256
                
                # Add distinct geometric patterns based on image index
                pattern_type = i % 4
                if pattern_type == 0:
                    # Central square pattern
                    if 64 <= x < 192 and 64 <= y < 192:
                        r = 255
                        g = (i * 10) % 256
                        b = 50
                elif pattern_type == 1:
                    # Diagonal stripe
                    if abs(x - y) < 15:
                        r = 30
                        g = 220
                        b = (i * 20) % 256
                elif pattern_type == 2:
                    # Central circle
                    cx, cy = 128, 128
                    if (x - cx)**2 + (y - cy)**2 < 60**2:
                        r = (i * 15) % 256
                        g = 40
                        b = 200
                elif pattern_type == 3:
                    # Horizontal grid lines
                    if y % 32 < 4:
                        r = 20
                        g = 20
                        b = 255
                
                pixels[idx] = r & 0xFF
                pixels[idx+1] = g & 0xFF
                pixels[idx+2] = b & 0xFF
                idx += 3
                
        # Write PPM P6 file
        with open(filename, "wb") as f:
            header = f"P6\n{width} {height}\n255\n"
            f.write(header.encode("ascii"))
            f.write(pixels)
            
    print(f"Successfully generated {num_images} images.")

if __name__ == "__main__":
    main()

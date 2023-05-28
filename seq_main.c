#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
    
    int width = 0, height = 0, channels = 0;
    
    // Load the image file
    unsigned char *input_image = stbi_load(argv[1], &width, &height, &channels, STBI_rgb);

    if (input_image == NULL) {
        fprintf(stderr, "Error loading input image.\n");
        return 1;
    }
    
    // Ensure the dimensions are divisible by 2
    if (width % 2 != 0 || height % 2 != 0) {
        fprintf(stderr, "Both width and height should be divisible by 2.\n");
        return 1;
    }
    
    int new_width = width / 2;
    int new_height = height / 2;
    
    // Allocate memory for the output image
    unsigned char *output_image = (unsigned char *)calloc(new_width * new_height, channels);
    
    if (output_image == NULL) {
        fprintf(stderr, "Memory allocation for output_image failed.\n");
        return 1;
    }
    
    clock_t start = clock();
    int result = stbir_resize_uint8(input_image, width, height, 0, output_image, new_width, new_height, 0, channels);
    clock_t end = clock();
    
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %lf seconds\n", elapsed_time);

    if (result) {
        printf("Resize operation was successful.\n");
    } else {
        fprintf(stderr, "Resize operation failed.\n");
        free(output_image);
        stbi_image_free(input_image);
        return 1;
    }

    if (!stbi_write_png(argv[2], new_width, new_height, channels, output_image, new_width * channels)) {
        fprintf(stderr, "Failed to write image\n");
        free(output_image);
        stbi_image_free(input_image);
        return 1;
    }

    free(output_image);
    stbi_image_free(input_image);
    return 0;
}

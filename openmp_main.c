#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void downsample_image(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels) {
    int new_width = width / 2;
    int new_height = height / 2;
    #pragma omp parallel for default(none) shared(input_image, output_image, width, height, channels, new_width, new_height)
    for (int i = 0; i < new_height; ++i) {
        for (int j = 0; j < new_width; ++j) {
            for (int k = 0; k < channels; ++k) {
                // Compute the fractional coordinates
                float x = j * 2.0f;
                float y = i * 2.0f;
                
                // Get the four surrounding pixels
                int x0 = (int)x;
                int x1 = x0 + 1;
                int y0 = (int)y;
                int y1 = y0 + 1;
                
                // Compute the fractional parts
                float dx = x - x0;
                float dy = y - y0;
                
                // Perform bilinear interpolation
                float value = (1.0f - dx) * (1.0f - dy) * input_image[(y0 * width + x0) * channels + k] +
                              dx * (1.0f - dy) * input_image[(y0 * width + x1) * channels + k] +
                              (1.0f - dx) * dy * input_image[(y1 * width + x0) * channels + k] +
                              dx * dy * input_image[(y1 * width + x1) * channels + k];
                
                // Assign the interpolated value to the output image
                output_image[(i * new_width + j) * channels + k] = (unsigned char)value;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Ensure we have enough arguments
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image.jpg> <output_image.png> <num_threads>\n", argv[0]);
        return 1;
    }

    // Parse the command line arguments
    char *input_image_filename = argv[1];
    char *output_image_filename = argv[2];
    int num_threads = atoi(argv[3]);

    // Load the image file
    int width, height, channels;
    unsigned char *input_image = stbi_load(input_image_filename, &width, &height, &channels, 0);
    if (input_image == NULL) {
        fprintf(stderr, "Could not load image file %s.\n", input_image_filename);
        return 1;
    }

    // Print the parsed values to check them
    printf("Input image: %s\n", input_image_filename);
    printf("Output image: %s\n", output_image_filename);
    printf("Number of threads: %d\n", num_threads);
    printf("Image dimensions: %d x %d\n", width, height);
    printf("Channels: %d\n", channels);

    // Calculate new dimensions and allocate memory for the output image
    int new_width = width / 2;
    int new_height = height / 2;
    unsigned char *output_image = malloc(new_width * new_height * channels);

    // Set the number of threads
    omp_set_num_threads(num_threads);

    // Perform the downsampling operation using different scheduling options
    char *schedules[] = {"default", "static,1", "static,100", "dynamic,1", "dynamic,100", "guided,100", "guided,1000"};
    int num_schedules = sizeof(schedules) / sizeof(schedules[0]);

    for (int i = 0; i < num_schedules; ++i) {
        double start_time, end_time, elapsed_time;

        start_time = omp_get_wtime();
        #pragma omp parallel for default(none) shared(input_image, output_image, width, height, channels, new_width, new_height) schedule(runtime)
        for (int i = 0; i < new_height; ++i) {
            downsample_image(input_image, output_image, width, height, channels);
        }
        end_time = omp_get_wtime();
        elapsed_time = end_time - start_time;
        printf("Elapsed time with schedule %s: %f seconds\n", schedules[i], elapsed_time);
    }

    // Write the downscaled image to the file
    stbi_write_png(output_image_filename, new_width, new_height, channels, output_image, new_width * channels);

    // Free the image data
    stbi_image_free(input_image);
    free(output_image);

    return 0;
}

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void downsample_image(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int start_height, int end_height);

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned char *input_image, *output_image;
    int width, height, channels;

    // Start recording the time
    start_time = MPI_Wtime();

    int rows_per_process, remainder_rows;
    
    if(rank == 0) {
        // Load the input image using stb_image
        input_image = stbi_load(argv[1], &width, &height, &channels, 0);

        // Calculate rows per process and remainder rows
        rows_per_process = height / size;
        remainder_rows = height % size;

        // Calculate the total number of rows for the output image
        int total_rows = (height / 2) + remainder_rows;

        // Allocate space for the downscaled image
        output_image = malloc((width / 2) * ((height / 2) + remainder_rows) * channels);

    }

    // Send the image size information to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&remainder_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute the image data among processes using MPI_Scatter
    int chunk_size = width * rows_per_process * channels;
    unsigned char *chunk_data = malloc(chunk_size + (rank < remainder_rows ? width * channels : 0));
    MPI_Scatter(input_image, chunk_size, MPI_UNSIGNED_CHAR, chunk_data, chunk_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Initialize OpenMP with the number of threads provided
    int num_threads = atoi(argv[3]);
    unsigned char *downscaled_chunk_data = malloc((width / 2) * (rows_per_process / 2) * channels + (rank < remainder_rows ? (width / 2) * channels : 0));

    #pragma omp parallel num_threads(num_threads)
    {
        // This should only process chunk_data, not the whole input_image
        downsample_image(chunk_data, downscaled_chunk_data, width, rows_per_process + (rank < remainder_rows ? 1 : 0), channels, 0, rank * rows_per_process);
    }

    // Collect downscaled portions from all processes using MPI_Gather
    int gather_size = (width / 2) * ((rows_per_process / 2) + (rank < remainder_rows ? 1 : 0)) * channels;
    int gather_offset = (rank * (width / 2) * (rows_per_process / 2) * channels) + (rank < remainder_rows ? (width / 2) * channels : 0);

    MPI_Gather(downscaled_chunk_data, gather_size + (rank < remainder_rows ? (width / 2) * channels : 0), MPI_UNSIGNED_CHAR, output_image + gather_offset, gather_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        // Write the downscaled image to the output file using stb_image_write
        stbi_write_jpg(argv[2], width / 2, (height / 2) + remainder_rows, channels, output_image, 100);

        // Free the memory for the images
        stbi_image_free(input_image);
        free(output_image);
    }

    // End recording the time
    end_time = MPI_Wtime();

    // Print out the elapsed time
    if (rank == 0) {
        printf("Time elapsed is %f seconds\n", end_time - start_time);
    }

    // Free the allocated memory
    free(chunk_data);
    free(downscaled_chunk_data);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}

void downsample_image(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int start_height, int end_height) {
    int new_width = width / 2;
    int new_height = height / 2;

    #pragma omp parallel for collapse(2)
    for (int i = start_height; i < end_height; i += 2) {
        for (int j = 0; j < new_width; j += 2) {
            for (int k = 0; k < channels; ++k) {
                int in_index = ((i - start_height) * width + j * 2) * channels + k;
                int out_index = ((i - start_height) / 2 * new_width + (j / 2)) * channels + k;

                output_image[out_index] = (input_image[in_index] +
                                           input_image[in_index + channels] +
                                           input_image[in_index + width * channels] +
                                           input_image[in_index + (width + 1) * channels]) / 4;
            }
        }
    }
}

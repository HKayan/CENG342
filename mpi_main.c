#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <hwloc.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

hwloc_topology_t topology;
hwloc_cpuset_t set_performance[8];
hwloc_cpuset_t set_efficiency[8];


void downsample_image(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels);
double perform_downsampling(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int num_threads);

int main(int argc, char* argv[]) {

    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // Allocate and set the hwloc_cpuset_t objects
    for (int i = 0; i < 8; ++i) {
        set_performance[i] = hwloc_bitmap_alloc();
        hwloc_bitmap_only(set_performance[i], i);

        set_efficiency[i] = hwloc_bitmap_alloc();
        hwloc_bitmap_only(set_efficiency[i], i+8);
    }
    // Ensure we have enough arguments
    
    // Parse the command line arguments
    char *input_image_filename = argv[1];
    char *output_image_filename = argv[2];
    int num_threads = atoi(argv[3]);
    
    double start_program = MPI_Wtime();

    double start_resize_time, end_resize_time;
    double local_resize_time;
    double max_local_resize_time = 0.0;

    int width = 0, height = 0, channels = 3;
    int rank = 0, size = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned char *input_image = NULL;

    if (rank == 0) {
        input_image = stbi_load(argv[1], &width, &height, &channels, STBI_rgb);
        if (input_image == NULL) {
            printf("File couldn't be loaded");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Print the parsed values to check them
        printf("Input image: %s\n", input_image_filename);
        printf("Output image: %s\n", output_image_filename);
        printf("Number of threads: %d\n", num_threads);
        printf("Image dimensions: %d x %d\n", width, height);
        printf("Channels: %d\n", channels);
    }

    

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block_size = width / size;
    int block_height = height;

    unsigned char *input_block = (unsigned char *)malloc(block_size * block_height * channels);
    unsigned char *output_block = (unsigned char *)malloc(block_size / 2 * block_height / 2 * channels);

    MPI_Scatter(input_image, block_size * block_height * channels, MPI_UNSIGNED_CHAR,
                input_block, block_size * block_height * channels, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);


    local_resize_time = perform_downsampling(input_block, output_block, block_size, block_height, channels, num_threads);

    MPI_Reduce(&local_resize_time, &max_local_resize_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max local resize time: %f seconds\n", max_local_resize_time);
    }

    MPI_Gather(output_block, block_size / 2 * block_height / 2 * channels, MPI_UNSIGNED_CHAR,
               input_image, block_size / 2 * block_height / 2 * channels, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        stbi_write_png(argv[2], width / 2, height / 2, channels, input_image, width / 2 * channels);
    }

    free(input_block);
    free(output_block);

    for (int i = 0; i < 8; ++i) {
        hwloc_bitmap_free(set_performance[i]);
        hwloc_bitmap_free(set_efficiency[i]);
    }

    hwloc_topology_destroy(topology);
    

    MPI_Finalize();

    if (rank == 0) {
        printf("The program was completed in %f seconds.\n", MPI_Wtime() - start_program);
    }

    return 0;
}

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

double perform_downsampling(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int num_threads) {
    
    omp_set_num_threads(8);
    omp_set_schedule(omp_sched_dynamic, 1);
    
    double start_time, end_time;
    double total_time = 0.0;

    #pragma omp parallel reduction(+: total_time)
    {
        int tid = omp_get_thread_num();
        if (tid < 8) {
            // bind to performance core
            hwloc_set_cpubind(topology, set_performance[tid], 0);
        } else {
            // bind to efficiency core
            hwloc_set_cpubind(topology, set_efficiency[tid-8], 0);
        }

        start_time = omp_get_wtime();
        downsample_image(input_image, output_image, width, height, channels);
        end_time = omp_get_wtime();
        total_time += (end_time - start_time);
    }

    double average_time = total_time / num_threads;

    return average_time;
}

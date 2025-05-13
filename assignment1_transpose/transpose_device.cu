#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"
#include <stdio.h>
#include <cuda.h>

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */



#define BS 32
#define PAD 1 // using padding to avoid bank conflicts

__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

   // non-coalesced global memory write, each of the thread write in a row major order
    output[j + n * i] = input[i + n * j];

}


__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

     __shared__ float datatile[BS][BS + PAD]; // padding

    // Calculate global indices for reading the input.
    int i = blockIdx.x * BS + threadIdx.x;
    int j = blockIdx.y * BS + threadIdx.y;

    // Coalesced global memory read:
    datatile[threadIdx.y][threadIdx.x] = input[j * n + i]; // r

    __syncthreads();

    // Calculate transposed indices.
    i = blockIdx.y * BS + threadIdx.x;  // Swap block indices for transposed write.
    j = blockIdx.x * BS + threadIdx.y;

    // Coalesced global memory write: threads write consecutive memory locations.
    output[j * n + i] = datatile[threadIdx.x][threadIdx.y];

}


#define TILE_DIM 32
#define BLOCK_ROWS 32
__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling (thread coarsening)
    __shared__ float tile[TILE_DIM][TILE_DIM+1];  // 64x65: padding prevents bank conflicts

        // Calculate the starting indices for this 64x64 tile.
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Each thread loads multiple elements from global memory into shared memory.
    // The loop is simple (not unrolled) to keep things beginner-friendly.
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // Coalesced global memory read.
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * n + x];
    }
    __syncthreads();

    // Compute transposed indices for writing.
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // Swap block indices for output.
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // Coalesced global memory write.
        output[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    // you can change the block dims
    if (type == NAIVE) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(32, 32);
        dim3 gridSize(n / 32, n / 32);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}


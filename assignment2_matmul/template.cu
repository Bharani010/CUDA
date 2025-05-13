#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A: m x k, B: k x n, C: m x n (row-major)

// Kernel 1: naive row-indexing 
__global__ void matmul1_naive(float *A, float *B, float *C, int M, int N, int K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // column index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// Kernel 2: coalesced indexing 
__global__ void matmul2_coalesced(float *A, float *B, float *C, int M, int N, int K) {
    int j = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // row index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// Kernel 3: 2x2 coarsened with shared memory
//32x32 shared memory tiles to reduce global memory access; each thread computes a 2x2 block (coarsening) to increase computing
#define TS3 16
__global__ void coarsened_matmul2x2(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TS3 * 2][TS3 * 2]; // 32x32 tile  A
    __shared__ float Bs[TS3 * 2][TS3 * 2]; // 32x32 tile  B
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each block computes a 32x32 tile; each thread a 2x2 block
    int row_base = by * (TS3 * 2);  // 32 rows per block
    int col_base = bx * (TS3 * 2);  // 32 cols per block
    int row = row_base + ty * 2;    // Starting row of 2x2 block
    int col = col_base + tx * 2;    // Starting col of 2x2 block
    
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    
    for (int t = 0; t < (K + TS3 * 2 - 1) / (TS3 * 2); t++) {
        // Load 32x32 tiles into shared memory
        int aRow = row_base + ty * 2;  // Load 2 elements per thread in y
        int aCol = t * (TS3 * 2) + tx * 2;
        int bRow = t * (TS3 * 2) + ty * 2;
        int bCol = col_base + tx * 2;
        
        // Load 2x2 block for A (coalesced)
        As[ty * 2][tx * 2] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        As[ty * 2 + 1][tx * 2] = (aRow + 1 < M && aCol < K) ? A[(aRow + 1) * K + aCol] : 0.0f;
        As[ty * 2][tx * 2 + 1] = (aRow < M && aCol + 1 < K) ? A[aRow * K + aCol + 1] : 0.0f;
        As[ty * 2 + 1][tx * 2 + 1] = (aRow + 1 < M && aCol + 1 < K) ? A[(aRow + 1) * K + aCol + 1] : 0.0f;
        
        // Load 2x2 block for B
        Bs[ty * 2][tx * 2] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        Bs[ty * 2 + 1][tx * 2] = (bRow + 1 < K && bCol < N) ? B[(bRow + 1) * N + bCol] : 0.0f;
        Bs[ty * 2][tx * 2 + 1] = (bRow < K && bCol + 1 < N) ? B[bRow * N + bCol + 1] : 0.0f;
        Bs[ty * 2 + 1][tx * 2 + 1] = (bRow + 1 < K && bCol + 1 < N) ? B[(bRow + 1) * N + bCol + 1] : 0.0f;
        
        __syncthreads();
        
        // Compute 2x2 block
        for (int k = 0; k < TS3 * 2; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }
    
    // Write 2x2 block to C
    if (row < M && col < N) C[row * N + col] = c00;
    if (row < M && col + 1 < N) C[row * N + col + 1] = c01;
    if (row + 1 < M && col < N) C[(row + 1) * N + col] = c10;
    if (row + 1 < M && col + 1 < N) C[(row + 1) * N + col + 1] = c11;
}

// Kernel 4: Shared memory tiled with 32x32 tiles 
//32x32 shared memory tiles for better data reuse; loop unrolling reduces overhead in the inner computation loop.
#define TS4 32
__global__ void MatMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TS4][TS4];
    __shared__ float Bs[TS4][TS4];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TS4 + ty;
    int col = bx * TS4 + tx;
    float c = 0.0f;
    
    for (int t = 0; t < (K + TS4 - 1) / TS4; t++) {
        if (row < M && (t * TS4 + tx) < K)
            As[ty][tx] = A[row * K + t * TS4 + tx];
        else
            As[ty][tx] = 0.0f;
        
        if (col < N && (t * TS4 + ty) < K)
            Bs[ty][tx] = B[(t * TS4 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll // reduces loop overhead 
        for (int k = 0; k < TS4; k++)
            c += As[ty][k] * Bs[k][tx];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = c;
}

// Kernel 5: Best optimized with 4x4 coarsening 
// Uses 64x16 and 16x64 shared memory tiles for high data reuse; 4x4 coarsening per thread and loop unrolling maximize compute intensity.
#define BS 16
__global__ void MatmulBest(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[64][BS];  // 64 rows x 16 cols
    __shared__ float Bs[BS][64];  // 16 rows x 64 cols
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_start = by * 64 + ty * 4;
    int col_start = bx * 64 + tx * 4;
    float cval[4][4] = {{0}};
    
    for (int t = 0; t < (K + BS - 1) / BS; t++) {
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int aRow = ty + 16 * p;
            int globalRow = by * 64 + aRow;
            int globalCol = t * BS + tx;
            As[aRow][tx] = (aRow < 64 && globalRow < M && globalCol < K) ?
                           A[globalRow * K + globalCol] : 0.0f;
        }
        
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            int bCol = tx + 16 * q;
            int globalRow = t * BS + ty;
            int globalCol = bx * 64 + bCol;
            Bs[ty][bCol] = (bCol < 64 && globalRow < K && globalCol < N) ?
                           B[globalRow * N + globalCol] : 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < BS; k++) {
                    sum += As[ty * 4 + i][k] * Bs[k][tx * 4 + j];
                }
                cval[i][j] += sum;
            }
        }
        
        __syncthreads();
    }
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = row_start + i;
            int c = col_start + j;
            if (r < M && c < N) {
                C[r * N + c] = cval[i][j];
            }
        }
    }
}

// Kernel launcher function
extern "C" void launchMatMulKernel(int kernelId, float *A, float *B, float *C, int M, int N, int K) {
    if (kernelId == 1) {
        dim3 blockDim(16, 16); 
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 2) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        matmul2_coalesced<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 3) {
        dim3 blockDim(TS3, TS3);  // 16x16 threads
        dim3 gridDim((N + TS3 * 2 - 1) / (TS3 * 2), (M + TS3 * 2 - 1) / (TS3 * 2)); // 32x32 tiles
        coarsened_matmul2x2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 4) {
        dim3 blockDim(TS4, TS4);
        dim3 gridDim((N + TS4 - 1) / TS4, (M + TS4 - 1) / TS4);
        MatMulTiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 5) {
        dim3 blockDim(BS, BS);
        dim3 gridDim((N + BS * 4 - 1) / (BS * 4), (M + BS * 4 - 1) / (BS * 4)); // 64x64 tiles
        MatmulBest<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else {
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    cudaDeviceSynchronize();
}
// CUDA headers first to avoid conflicts with STL macros
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

// Thrust headers for sorting and parallel operations
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

// Standard C++ headers after CUDA
#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

// Assignment-specific headers
#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

// CUDA Error Checking Macro (provided by the assignment)
// this ensures that CUDA API calls are checked for errors at development
#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// Fireworks constants
#define NUM_FIREWORKS 15
#define NUM_SPARKS 20

// Static variables for tiling data
// these hold the tiling info globally instead of as class members
static int* static_cudaDevicePairsTileIdx = nullptr;
static int* static_cudaDevicePairsCircleIdx = nullptr;
static int* static_cudaDeviceNumPairs = nullptr;
static int* static_cudaDeviceTileStarts = nullptr;
static int* static_cudaDeviceTileLengths = nullptr;
static int static_hostMaxPairs = 0;
static int static_hostNumTiles = 0;

// Structure to hold global constants that are copied to CUDA constant memory
// These constants are accessible by all CUDA kernels
struct GlobalConstants {
    SceneName sceneName;          
    int numCircles;               
    float* position;              
    float* velocity;              
    float* color;                
    float* radius;                
    int imageWidth;               
    int imageHeight;              
    float* imageData;             
    int tileSize;                 
    int numTilesX;                
    int numTilesY;                
    int numTiles;                 
    int* pairsTileIdx;            
    int* pairsCircleIdx;          
    int* numPairs;                
    int* tileStarts;              
    int* tileLengths;             
    int maxPairs;                 
};

// Constant memory variable to store global constants
__constant__ GlobalConstants cuConstRendererParams;

// Constant mem for noise tables for the snowflakes scene
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// Constant mem for color lookup table for the snowflakes scene
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// Include inline CUDA files after all headers
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// Sets each pixel to a shade of gray based on its y-position to simulate a gradient background
__global__ void kernelClearImageSnowflake() {
    // Compute the pixel coordinates for this thread
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    // edge case if pixel is outside the image bounds, skips
    if (imageX >= width || imageY >= height) return;

    // Compute the offset into the image data array
    int offset = 4 * (imageY * width + imageX);

    // Compute a shade value that varies with y-position 
    float shade = 0.4f + 0.45f * static_cast<float>(height - imageY) / height;

    // Set the pixel color to a shade of gray with full opacity
    float4 value = make_float4(shade, shade, shade, 1.f);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// Kernel to clear the image with a solid color
// Used for all scenes except snowflakes
__global__ void kernelClearImage(float r, float g, float b, float a) {
    // Compute the pixel coordinates for this thread
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;
    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    // edge case if pixel is outside the image bounds, skips
    if (imageX >= width || imageY >= height) return;

    // Compute the offset into the image data array
    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// Kernel to advance the animation for the fireworks scene
// Updates the positions and velocities of spark particles for each firework
__global__ void kernelAdvanceFireWorks() {
    // Constants for the simulation
    const float dt = 1.f / 60.f;        // Time step (1/60 seconds for 60 FPS)
    const float pi = 3.14159;           // Pi constant for angular calculations
    const float maxDist = 0.25f;        // Maximum distance sparks can travel before resetting
    const float gravity = -9.8f;        // Gravity acceleration
    const float dragCoeff = 0.95f;      // Drag coefficient to slow down sparks

    // Access device pointers to velocity, position, and radius arrays
    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    // Compute the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles || index < NUM_FIREWORKS) return;

    // indices for the firework and sparks
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;        // Firework index
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;       // Spark index within the firework
    int index3i = 3 * fIdx;                                 // 3D index for the firework
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;   // Spark index
    int index3j = 3 * sIdx;                                 // 3D index for the spark

    // Get the firework center and spark position
    float cx = position[index3i];
    float cy = position[index3i + 1];
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // distance between the spark and the firework center
    float cxsx = sx - cx;
    float cysy = sy - cy;
    float dist = sqrt(cxsx * cxsx + cysy * cysy);

    // If the spark has traveled too far, reset its position and velocity
    if (dist > maxDist) {
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        // Reset the spark position relative to the firework center
        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // Set the initial velocity of the spark (radial outward)
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    } else {
        velocity[index3j + 1] += gravity * dt;
        velocity[index3j] *= dragCoeff;
        velocity[index3j + 1] *= dragCoeff;
        position[index3j] += velocity[index3j] * dt;
        position[index3j + 1] += velocity[index3j + 1] * dt;
        position[index3j + 2] += velocity[index3j + 2] * dt;
    }
}

// Kernel to advance the animation for the hypnosis scene
// Updates the radius of each circle to create a pulsating effect
__global__ void kernelAdvanceHypnosis() {
    // global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    // radius array
    float* radius = cuConstRendererParams.radius;
    float cutOff = 0.5f;    // Maximum radius before reset

    // If the radius exceeds the cutoff, reset it; if not, increase it
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}

// Kernel to advance the animation for the bouncing balls scene
// Updates the positions and velocities of balls under gravity with bouncing behavior
__global__ void kernelAdvanceBouncingBalls() {
    // Constants for the simulation
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f;
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    // velocity and position arrays
    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    int index3 = 3 * index;
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    // Skip if the ball is already at rest
    if (oldVelocity == 0.f && oldPosition == 0.f) return;

    // applying drag to make the ball bounce
    if (position[index3 + 1] < 0 && oldVelocity < 0.f) {
        velocity[index3 + 1] *= kDragCoeff;
    }

    velocity[index3 + 1] += kGravity * dt;
    position[index3 + 1] += velocity[index3 + 1] * dt;

    // Check if the ball has stopped 
    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon &&
        oldPosition < 0.0f &&
        fabsf(position[index3 + 1] - oldPosition) < epsilon) {
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// Kernel to advance the animation for the snowflake scene
// Updates positions and velocities of snowflakes with noise-based forces
__global__ void kernelAdvanceSnowflake() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    // Constants for the simulation
    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f;
    const float kDragCoeff = 2.f;
    int index3 = 3 * index;
    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f);

    // Compute noise forces using a noise function
    float3 noiseInput = {10.f * position.x, 10.f * position.y, 255.f * position.z};
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;
    float2 dragForce = {-kDragCoeff * velocity.x, -kDragCoeff * velocity.y};
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    velocity.x += forceScaling * (noiseForce.x + dragForce.x) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    // Get the radius of the snowflake
    float radius = cuConstRendererParams.radius[index];

    // If the snowflake moves out of bounds, reset its position
    if ((position.y + radius < 0.f) || (position.x + radius < -0.f) || (position.x - radius > 1.f)) {
        noiseInput = {255.f * position.x, 255.f * position.y, 255.f * position.z};
        float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store the updated position and velocity
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// Kernel to build (tile, circle) pairs
// Each thread processes one circle and determines which tiles it intersects
__global__ void kernelBuildPairs() {
    // Compute the global thread index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) return;

    // Get the circle's position and radius
    int index3 = 3 * index;
    float3 p = make_float3(cuConstRendererParams.position[index3],
                           cuConstRendererParams.position[index3 + 1],
                           cuConstRendererParams.position[index3 + 2]);
    float rad = cuConstRendererParams.radius[index];

    // Get image dimensions and tiling parameters
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int tileSize = cuConstRendererParams.tileSize;
    int numTilesX = cuConstRendererParams.numTilesX;
    int numTilesY = cuConstRendererParams.numTilesY;

    // Compute the bounding box of the circle in normalized coordinates [0, 1]
    float minX = fmaxf(p.x - rad, 0.f);
    float maxX = fminf(p.x + rad, 1.f);
    float minY = fmaxf(p.y - rad, 0.f);
    float maxY = fminf(p.y + rad, 1.f);

    // Convert to pixel coordinates
    int pixelMinX = floor(minX * imageWidth);
    int pixelMaxX = ceil(maxX * imageWidth) - 1;
    int pixelMinY = floor(minY * imageHeight);
    int pixelMaxY = ceil(maxY * imageHeight) - 1;

    // Compute the range of tiles that the circle intersects
    int tileMinX = max(pixelMinX / tileSize, 0);
    int tileMaxX = min(pixelMaxX / tileSize, numTilesX - 1);
    int tileMinY = max(pixelMinY / tileSize, 0);
    int tileMaxY = min(pixelMaxY / tileSize, numTilesY - 1);

    // For each intersecting tile, create a (tile, circle) pair
    for (int ty = tileMinY; ty <= tileMaxY; ty++) {
        for (int tx = tileMinX; tx <= tileMaxX; tx++) {
            int tileIdx = ty * numTilesX + tx;
            // Atomically increment the number of pairs and get the index to store the pair
            int idx = atomicAdd(cuConstRendererParams.numPairs, 1);
            if (idx < cuConstRendererParams.maxPairs) {
                cuConstRendererParams.pairsTileIdx[idx] = tileIdx;
                cuConstRendererParams.pairsCircleIdx[idx] = index;
            }
        }
    }
}

// Kernel to render pixels using the tiling approach
// Each block processes one tile, and threads within the block process pixels in the tile
__global__ void kernelRenderPixels() {
    // Get the tile index for this block
    int tileIdx = blockIdx.x;
    if (tileIdx >= cuConstRendererParams.numTiles) return;

    // Get the starting index and length of the (tile, circle) pairs for this tile
    int start = cuConstRendererParams.tileStarts[tileIdx];
    int length = cuConstRendererParams.tileLengths[tileIdx];
    if (length == 0) return;

    int tileX = tileIdx % cuConstRendererParams.numTilesX;
    int tileY = tileIdx / cuConstRendererParams.numTilesX;
    int tileSize = cuConstRendererParams.tileSize;
    int pixelX = tileX * tileSize + threadIdx.x;
    int pixelY = tileY * tileSize + threadIdx.y;
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;

    if (pixelX >= imageWidth || pixelY >= imageHeight) return;

    int offset = 4 * (pixelY * imageWidth + pixelX);
    float4 currentColor = *(float4*)(&cuConstRendererParams.imageData[offset]);
    float2 pixelCenter = make_float2((pixelX + 0.5f) / imageWidth, (pixelY + 0.5f) / imageHeight);

    for (int i = 0; i < length; i++) {
        int circleIdx = cuConstRendererParams.pairsCircleIdx[start + i];
        int index3 = 3 * circleIdx;
        float3 p = make_float3(cuConstRendererParams.position[index3],
                               cuConstRendererParams.position[index3 + 1],
                               cuConstRendererParams.position[index3 + 2]);
        float rad = cuConstRendererParams.radius[circleIdx];

        // Compute the distance from the pixel center to the circle center
        float diffX = p.x - pixelCenter.x;
        float diffY = p.y - pixelCenter.y;
        float pixelDist = diffX * diffX + diffY * diffY;

        // If the pixel center is inside the circle, blend the circle's color
        if (pixelDist <= rad * rad) {
            float3 rgb;
            float alpha;

            // Special handling for snowflake scenes (use a color lookup and depth-based alpha)
            if (cuConstRendererParams.sceneName == SNOWFLAKES ||
                cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                float normPixelDist = sqrt(pixelDist) / rad;
                rgb = lookupColor(normPixelDist);   // Get color from lookup table
                const float kCircleMaxAlpha = 0.5f;
                const float falloffScale = 4.f;
                float maxAlpha = 0.6f + 0.4f * (1.f - p.z); // Alpha varies with depth
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f);
                alpha = maxAlpha * exp(-falloffScale * normPixelDist * normPixelDist);
            } else {
                // For other scenes, use the circle's color and a fixed alpha
                rgb = *(float3*)(&cuConstRendererParams.color[index3]);
                alpha = 0.5f;
            }

            // Blend the circle's color with the current pixel color
            float oneMinusAlpha = 1.f - alpha;
            currentColor.x = alpha * rgb.x + oneMinusAlpha * currentColor.x;
            currentColor.y = alpha * rgb.y + oneMinusAlpha * currentColor.y;
            currentColor.z = alpha * rgb.z + oneMinusAlpha * currentColor.z;
            currentColor.w = alpha + currentColor.w;    // Update alpha channel
        }
    }

    // Write the final color back to the image
    *(float4*)(&cuConstRendererParams.imageData[offset]) = currentColor;
}

// Constructor for CudaRenderer
// Initializes member variables to NULL
CudaRenderer::CudaRenderer() {
    image = NULL;
    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;
    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

// Destructor for CudaRenderer
// Frees all allocated memory
CudaRenderer::~CudaRenderer() {
    // Free host memory
    if (image) delete image;
    if (position) {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    // Free device memory
    if (cudaDevicePosition) {
        cudaCheckError(cudaFree(cudaDevicePosition));
        cudaCheckError(cudaFree(cudaDeviceVelocity));
        cudaCheckError(cudaFree(cudaDeviceColor));
        cudaCheckError(cudaFree(cudaDeviceRadius));
        cudaCheckError(cudaFree(cudaDeviceImageData));
    }

    // Free static tiling data memory
    if (static_cudaDevicePairsTileIdx) {
        cudaCheckError(cudaFree(static_cudaDevicePairsTileIdx));
        static_cudaDevicePairsTileIdx = nullptr;
    }
    if (static_cudaDevicePairsCircleIdx) {
        cudaCheckError(cudaFree(static_cudaDevicePairsCircleIdx));
        static_cudaDevicePairsCircleIdx = nullptr;
    }
    if (static_cudaDeviceNumPairs) {
        cudaCheckError(cudaFree(static_cudaDeviceNumPairs));
        static_cudaDeviceNumPairs = nullptr;
    }
    if (static_cudaDeviceTileStarts) {
        cudaCheckError(cudaFree(static_cudaDeviceTileStarts));
        static_cudaDeviceTileStarts = nullptr;
    }
    if (static_cudaDeviceTileLengths) {
        cudaCheckError(cudaFree(static_cudaDeviceTileLengths));
        static_cudaDeviceTileLengths = nullptr;
    }
}

// Method to get the rendered image
// Copies the image data from the device to the host and returns the image
const Image* CudaRenderer::getImage() {
    printf("Copying image data from device\n");
    cudaCheckError(cudaMemcpy(image->data, cudaDeviceImageData,
                              sizeof(float) * 4 * image->width * image->height,
                              cudaMemcpyDeviceToHost));
    return image;
}

// Method to load a scene
// Loads the circle data for the specified scene
void CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

// Method to set up the renderer
// Allocates device memory, copies data to the device, and initializes constants
void CudaRenderer::setup() {
    // Print CUDA device information
    int deviceCount = 0;
    cudaCheckError(cudaGetDeviceCount(&deviceCount));
    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaCheckError(cudaGetDeviceProperties(&deviceProps, i));
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // Allocate device memory for circle data and image data
    cudaCheckError(cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles));
    cudaCheckError(cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height));

    // Copy circle data from host to device
    cudaCheckError(cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice));

    // Compute tiling parameters
    const int tileSize = 16;    // Tile size (16x16 pixels)
    int numTilesX = (image->width + tileSize - 1) / tileSize;   // Number of tiles along x-axis
    int numTilesY = (image->height + tileSize - 1) / tileSize;  // Number of tiles along y-axis
    int numTiles = numTilesX * numTilesY;                       // Total number of tiles

    // Compute the maximum number of (tile, circle) pairs to allocate memory for
    // This is done dynamically to avoid memory issues for large scenes
    const long long MAX_PAIR_MEMORY = 6LL * 1024 * 1024 * 1024; // 6 GB for pairs
    const long long maxPairsLimit = MAX_PAIR_MEMORY / (2 * sizeof(int)); // 2 arrays: pairsTileIdx and pairsCircleIdx
    int maxTilesPerCircle = maxPairsLimit / numCircles;
    maxTilesPerCircle = std::max(100, std::min(maxTilesPerCircle, 16000));
    int maxPairs = numCircles * maxTilesPerCircle;  // Total maximum pairs

    // Allocate device memory for tiling data structures
    cudaCheckError(cudaMalloc(&static_cudaDevicePairsTileIdx, sizeof(int) * maxPairs));
    cudaCheckError(cudaMalloc(&static_cudaDevicePairsCircleIdx, sizeof(int) * maxPairs));
    cudaCheckError(cudaMalloc(&static_cudaDeviceNumPairs, sizeof(int)));
    cudaCheckError(cudaMalloc(&static_cudaDeviceTileStarts, sizeof(int) * numTiles));
    cudaCheckError(cudaMalloc(&static_cudaDeviceTileLengths, sizeof(int) * numTiles));

    static_hostMaxPairs = maxPairs;
    static_hostNumTiles = numTiles;

    // Set up global constants
    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    params.tileSize = tileSize;
    params.numTilesX = numTilesX;
    params.numTilesY = numTilesY;
    params.numTiles = numTiles;
    params.pairsTileIdx = static_cudaDevicePairsTileIdx;
    params.pairsCircleIdx = static_cudaDevicePairsCircleIdx;
    params.numPairs = static_cudaDeviceNumPairs;
    params.tileStarts = static_cudaDeviceTileStarts;
    params.tileLengths = static_cudaDeviceTileLengths;
    params.maxPairs = maxPairs;

    // Copy global constants to constant memory
    cudaCheckError(cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants)));

    // Copy noise tables to constant memory (for snowflakes scene)
    int* permX, *permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256));
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256));
    cudaCheckError(cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256));

    // Copy color lookup table to constant memory (for snowflakes scene)
    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {.8f, .9f, 1.f}, {.8f, .9f, 1.f}, {.8f, 0.8f, 1.f}
    };
    cudaCheckError(cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE));
}

// Creates a new Image object with the specified dimensions
void CudaRenderer::allocOutputImage(int width, int height) {
    if (image) delete image;
    image = new Image(width, height);
}

// kernel to set the image to a background color
void CudaRenderer::clearImage() {
    // Set up the grid and block dimensions for the clear kernel
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x,
                 (image->height + blockDim.y - 1) / blockDim.y);

    // Use a gradient background for snowflake scenes, otherwise use a solid white background
    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaCheckError(cudaDeviceSynchronize());
}

void CudaRenderer::advanceAnimation() {
    static int frameCount = 0;
    printf("Advancing animation for scene %d, frame %d\n", sceneName, frameCount++);

    // Set up the grid and block dimensions for the animation kernel
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }

    // Synchronize to ensure the kernel completes
    cudaCheckError(cudaDeviceSynchronize());
}

void CudaRenderer::render() {
    cudaCheckError(cudaMemset(static_cudaDeviceNumPairs, 0, sizeof(int)));

    // Set up grid and block for building pairs
    dim3 blockDimBuild(256, 1);
    dim3 gridDimBuild((numCircles + blockDimBuild.x - 1) / blockDimBuild.x);
    kernelBuildPairs<<<gridDimBuild, blockDimBuild>>>();
    cudaCheckError(cudaDeviceSynchronize());

    int numPairs;
    cudaCheckError(cudaMemcpy(&numPairs, static_cudaDeviceNumPairs, sizeof(int), cudaMemcpyDeviceToHost));

    // Check if the number of pairs exceeds allocated memory
    if (numPairs > static_hostMaxPairs) {
        printf("Warning: numPairs (%d) exceeds maxPairs (%d), clamping\n", numPairs, static_hostMaxPairs);
        numPairs = static_hostMaxPairs;
    }

    // Create Thrust device pointers for the pairs arrays
    thrust::device_ptr<int> dev_pairsTileIdx = thrust::device_pointer_cast(static_cudaDevicePairsTileIdx);
    thrust::device_ptr<int> dev_pairsCircleIdx = thrust::device_pointer_cast(static_cudaDevicePairsCircleIdx);

    // Sort the pairs by circle index, then by tile index
    thrust::sort_by_key(dev_pairsCircleIdx, dev_pairsCircleIdx + numPairs, dev_pairsTileIdx);
    thrust::stable_sort_by_key(dev_pairsTileIdx, dev_pairsTileIdx + numPairs, dev_pairsCircleIdx);

    // start indices and lengths of pairs for each tile
    int numTiles = static_hostNumTiles;
    thrust::device_vector<int> outputKeys(numTiles);
    thrust::device_vector<int> outputLengths(numTiles);
    thrust::constant_iterator<int> ones(1);
    auto newEnd = thrust::reduce_by_key(dev_pairsTileIdx, dev_pairsTileIdx + numPairs,
                                        ones,
                                        outputKeys.begin(),
                                        outputLengths.begin());
    int numUniqueTiles = newEnd.first - outputKeys.begin();

    thrust::device_vector<int> outputStarts(numUniqueTiles);
    thrust::exclusive_scan(outputLengths.begin(), outputLengths.begin() + numUniqueTiles, outputStarts.begin());

    std::vector<int> hostKeys(numUniqueTiles);
    std::vector<int> hostLengths(numUniqueTiles);
    std::vector<int> hostStarts(numUniqueTiles);
    thrust::copy(outputKeys.begin(), outputKeys.begin() + numUniqueTiles, hostKeys.begin());
    thrust::copy(outputLengths.begin(), outputLengths.begin() + numUniqueTiles, hostLengths.begin());
    thrust::copy(outputStarts.begin(), outputStarts.begin() + numUniqueTiles, hostStarts.begin());

    // Create arrays for tile starts and lengths, initializing unused tiles to -1 and 0
    std::vector<int> hostTileStarts(numTiles, -1);
    std::vector<int> hostTileLengths(numTiles, 0);
    for (int i = 0; i < numUniqueTiles; i++) {
        int tileIdx = hostKeys[i];
        hostTileStarts[tileIdx] = hostStarts[i];
        hostTileLengths[tileIdx] = hostLengths[i];
    }

    cudaCheckError(cudaMemcpy(static_cudaDeviceTileStarts, hostTileStarts.data(), sizeof(int) * numTiles, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(static_cudaDeviceTileLengths, hostTileLengths.data(), sizeof(int) * numTiles, cudaMemcpyHostToDevice));

    dim3 blockDimRender(16, 16);
    dim3 gridDimRender(numTiles);
    kernelRenderPixels<<<gridDimRender, blockDimRender>>>();
    cudaCheckError(cudaDeviceSynchronize());
}

// Method to shade a pixel 
void CudaRenderer::shadePixel(int circleIndex, float pixelCenterX, float pixelCenterY, float px, float py, float pz, float* pixelData) {
    // Not used in this optimized version
}
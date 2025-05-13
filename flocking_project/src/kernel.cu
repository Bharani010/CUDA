#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
/*below are the all the thrust libraries (c++ template) added for their specific functionality,
this makes it easier to move data between the CPU and the GPU.*/
#include <thrust/sort.h> // Required for thrust::sort_by_key
#include <thrust/device_vector.h> // Required for thrust::device_ptr
#include <thrust/functional.h> // Required for thrust::identity
#include <thrust/execution_policy.h> // Required for thrust execution policies (e.g., thrust::device)
#include <thrust/random.h> // Required for thrust random number generation
#include <thrust/host_vector.h> // Required for thrust::host_vector

#include "utilityCore.hpp"
#include "kernel.h"

// Toggle device-side printf spam
#ifndef DEBUG
#  define DEBUG 0
#endif

// Useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// Configuration for the boids algorithm.
#define rule1Distance 5.0f // Cohesion distance
#define rule2Distance 3.0f // Separation distance
#define rule3Distance 5.0f // Alignment distance

#define rule1Scale 0.01f // Cohesion force scale
#define rule2Scale 0.1f  // Separation force scale
#define rule3Scale 0.1f  // Alignment force scale

#define maxSpeed 1.0f    // Maximum boid speed

// Variable to control block size for performance testing
int Boids::currentBlockSize = blockSize;

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// Main buffers for boid information (corresponds to dev_pos, dev_vel1, dev_vel2 in reference)
glm::vec3 *d_boidPositions; // Main position buffer
glm::vec3 *d_velocitiesA;   // Main velocity buffer A (used as input/output ping-pong)
glm::vec3 *d_velocitiesB;   // Main velocity buffer B (used as input/output ping-pong)

// For efficient sorting and the uniform grid. These should always be parallel.
int *d_boidArrayIdx;         // Stores original index in main buffers for sorted data
int *d_boidGridIdx;          // Stores grid cell index for sorting

// Removed global thrust pointers - declare locally where needed

int *d_gridCellStart;        // Stores start index in sorted array for each grid cell
int *d_gridCellEnd;          // Stores end index in sorted array for each grid cell

// Additional buffers for coherent grid (corresponds to dev_coherentPos, dev_coherentVel in reference)
glm::vec3 *d_sortedPositions;   // Coherent position buffer
glm::vec3 *d_sortedVelocitiesA; // Coherent velocity buffer

// Removed surplus d_sortedVelocitiesB buffer declaration

// Grid parameters based on simulation parameters.
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
*****************/

__host__ __device__ unsigned int hash(unsigned int a) {
	// Restored original hash constants from reference
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}

/**
* Helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
	thrust::default_random_engine rng(hash((int)(index * time)));
	// Restored original random distribution range from reference
	thrust::uniform_real_distribution<float> unitDistrib(-1.0f, 1.0f);

	return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* CUDA kernel for generating boids with a random position somewhere inside the simulation space
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 rand = generateRandomVec3(time, index);
		arr[index].x = scale * rand.x;
		arr[index].y = scale * rand.y;
		arr[index].z = scale * rand.z;
	}
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
	numObjects = N;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	// Allocate basic boid buffers
	cudaMalloc((void**)&d_boidPositions, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc d_boidPositions failed!");

	cudaMalloc((void**)&d_velocitiesA, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc d_velocitiesA failed!");

	cudaMalloc((void**)&d_velocitiesB, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc d_velocitiesB failed!");

	// Initialize random positions
	kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
		d_boidPositions, scene_scale);
	checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

	// Compute grid parameters based on simulation parameters (Restored logic from reference)
	gridCellWidth = 0.5f * std::max({rule1Distance, rule2Distance, rule3Distance});
	int halfSideCount = (int)(scene_scale / gridCellWidth) + 1; // Add 1 for boundary cells
	gridSideCount = 2 * halfSideCount; // Double to cover both positive and negative sides from center

	gridCellCount = gridSideCount * gridSideCount * gridSideCount;
	gridInverseCellWidth = 1.0f / gridCellWidth;
	float halfGridWidth = gridCellWidth * halfSideCount;
	gridMinimum.x = -halfGridWidth;
	gridMinimum.y = -halfGridWidth;
	gridMinimum.z = -halfGridWidth;

	// Allocate additional buffers for grid-based approaches
	cudaMalloc((void**)&d_boidArrayIdx, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc d_boidArrayIdx failed!");

	cudaMalloc((void**)&d_boidGridIdx, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc d_boidGridIdx failed!");

	// Removed global thrust device pointer declarations - will wrap locally

	// Allocate cell start/end indices
	cudaMalloc((void**)&d_gridCellStart, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc d_gridCellStart failed!");

	cudaMalloc((void**)&d_gridCellEnd, gridCellCount * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc d_gridCellEnd failed!");

	// Allocate additional buffers for coherent grid approach
	cudaMalloc((void**)&d_sortedPositions, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc d_sortedPositions failed!");

	cudaMalloc((void**)&d_sortedVelocitiesA, N * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc d_sortedVelocitiesA failed!");

	// Removed cudaMalloc for d_sortedVelocitiesB

	cudaDeviceSynchronize(); // Keep final sync in init
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f; // Using velocities for color (0.3f offset for visibility)
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, d_boidPositions, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, d_velocitiesA, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyBoidsToVBO failed!");

	cudaDeviceSynchronize(); // Keep sync after VBO copy
}


/******************
* stepSimulation *
******************/

/**
* Helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute  new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 selfPos = pos[iSelf];
	glm::vec3 velChange(0.0f);

	glm::vec3 rule1Vector(0.0f); // Cohesion
	glm::vec3 rule2Vector(0.0f); // Separation
	glm::vec3 rule3Vector(0.0f); // Alignment

	int rule1Count = 0;
	int rule3Count = 0;

	for (int i = 0; i < N; ++i) {
		if (i == iSelf) continue; // Skip self

		glm::vec3 otherPos = pos[i];
		glm::vec3 diff = otherPos - selfPos;
		// Using glm::distance directly as requested
		float dist = glm::distance(otherPos, selfPos);

		// Rule 2: Separation - avoid crowding neighbors
		if (dist < rule2Distance) {
			rule2Vector -= diff; // Using diff directly
		}

		// Rule 1: Cohesion - fly towards the center of mass of neighbors
		if (dist < rule1Distance) {
			rule1Vector += otherPos;
			rule1Count++;
		}

		// Rule 3: Alignment - match velocity with nearby boids
		if (dist < rule3Distance) {
			rule3Vector += vel[i];
			rule3Count++;
		}
	}

	// Apply Rule 2 (Separation)
	velChange += rule2Vector * rule2Scale;

	// Apply Rule 1 (Cohesion) - fly towards center of mass of neighbors
	if (rule1Count > 0) {
		rule1Vector /= rule1Count;
		velChange += (rule1Vector - selfPos) * rule1Scale;
	}

	// Apply Rule 3 (Alignment) - match velocity with nearby boids
	if (rule3Count > 0) {
		rule3Vector /= rule3Count;
		velChange += rule3Vector * rule3Scale;
	}

	return velChange;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
	glm::vec3 *vel1, glm::vec3 *vel2) {
	// Get thread index
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Compute a new velocity based on pos and vel1
	glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);

	// Clamp the speed to maximum
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// Store the new velocity into vel2 for ping-pong buffering
	vel2[index] = newVel;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	glm::vec3 thisPos = pos[index];
	thisPos += vel[index] * dt;

	// Wrap the boids around so we don't lose them
	thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
	thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
	thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

	thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
	thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
	thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

	pos[index] = thisPos;
}

// Method of computing a 1D index from a 3D grid index.
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

/**
* Reset integer buffer to a specified value.
* Used to reset the grid cell start and end indices.
*/
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}
	intBuffer[index] = value;
}

/**
* Compute indices for boid array and grid cells.
*/
__global__ void kernComputeIndices(
	int N, int gridResolution,
	glm::vec3 gridMin, float inverseCellWidth,
	glm::vec3* pos, int* particleArrayIndices, int* particleGridIndices) {
	
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= N) {
		return;
	}

	// Store the particle's actual array index
	particleArrayIndices[index] = index;

	// Calculate the grid cell index for this particle
	glm::vec3 particlePos = pos[index];

	int gridX = (int)((particlePos.x - gridMin.x) * inverseCellWidth);
	int gridY = (int)((particlePos.y - gridMin.y) * inverseCellWidth);
	int gridZ = (int)((particlePos.z - gridMin.z) * inverseCellWidth);

	// Clamp grid indices to valid range
	gridX = max(0, min(gridX, gridResolution - 1));
	gridY = max(0, min(gridY, gridResolution - 1));
	gridZ = max(0, min(gridZ, gridResolution - 1));

	// Compute 1D grid cell index
	int gridIndex = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);

	// Store the grid cell index for this particle
	particleGridIndices[index] = gridIndex;
}

/**
* Identify the start and end points of each gridcell in the sorted particleGridIndices array.
* Corrected logic to match the reference implementation for setting end indices.
*/
__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
	int *gridCellStartIndices, int *gridCellEndIndices)
{
	//go through particleGridIndices identifying when there is a change in there value,
	//which signifies a change in the gridcell we are dealing with
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	// If this is the first particle or the previous particle belongs to a different cell
	if (index == 0 || particleGridIndices[index] != particleGridIndices[index - 1])
	{
		gridCellStartIndices[particleGridIndices[index]] = index;
	}

	// If this is the last particle or the next particle belongs to a different cell
	if (index == N - 1 || particleGridIndices[index] != particleGridIndices[index + 1])
	{
		gridCellEndIndices[particleGridIndices[index]] = index; // End index is inclusive
	}
}

/**
* Store the reshuffled position and velocity data in coherent buffers.
* Corresponds to kernSetCoherentPosVel in the reference.
*/
__global__ void kernRearrangeBoidData(
	int N, int *particleArrayIndices,
	const glm::vec3 *pos, const glm::vec3 *vel,
	glm::vec3 *coherentPos, glm::vec3 *coherentVel) // Renamed parameters to match purpose
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N)
	{
		return;
	}

	// Get the original boid index using the sorted array index
	int originalIndex = particleArrayIndices[index];

	// Rearrange positions and velocities to be coherent in memory
	// coherentPos[index] is the position of the index-th particle in the sorted list
	coherentPos[index] = pos[originalIndex];
	coherentVel[index] = vel[originalIndex]; // Copy velocity too for coherent access
}


/**
* Update a boid's velocity using the uniform grid to reduce neighbor search.
* Uses scattered memory access pattern for main data.
*/
__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	int *particleArrayIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// Get thread index
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	// Get current boid's position and velocity
	glm::vec3 boidPos = pos[index];
	glm::vec3 velChange(0.0f);

	// Calculate the grid cell this boid is in
	glm::vec3 relPos = boidPos - gridMin;
	int gridX = min(max(0, (int)(relPos.x * inverseCellWidth)), gridResolution - 1);
	int gridY = min(max(0, (int)(relPos.y * inverseCellWidth)), gridResolution - 1);
	int gridZ = min(max(0, (int)(relPos.z * inverseCellWidth)), gridResolution - 1);

	// Variables for collecting rule information
	glm::vec3 rule1Vector(0.0f); // Cohesion
	glm::vec3 rule2Vector(0.0f); // Separation
	glm::vec3 rule3Vector(0.0f); // Alignment
	int rule1Count = 0;
	int rule3Count = 0;

	// Reordered loops for X-Y-Z order as requested for scattered
	for (int i = -1; i <= 1; ++i) {                                         // X  (innermost)
		int neighborX = min(max(gridX + i, 0), gridResolution - 1);
		for (int j = -1; j <= 1; ++j) {                                     // Y  (middle)
			int neighborY = min(max(gridY + j, 0), gridResolution - 1);
			for (int k = -1; k <= 1; ++k) {                                 // Z  (outermost)
				int neighborZ = min(max(gridZ + k, 0), gridResolution - 1);

				int gridIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);

				// Get the start and end indices for this cell in the sorted array
				int cellStart = gridCellStartIndices[gridIndex];
				int cellEnd = gridCellEndIndices[gridIndex];

				// Skip empty cells (-1 indicates no particles)
				if (cellStart == -1) continue;

				// Iterate over all boids in this cell (using inclusive end index)
				for (int h = cellStart; h <= cellEnd; h++) { // Loop is <= cellEnd
					int boidIndex = particleArrayIndices[h]; // Access original index
					if (boidIndex == index) continue; // Skip self

					glm::vec3 otherPos = pos[boidIndex]; // Scattered access
					glm::vec3 diff = otherPos - boidPos;
					// Using glm::distance directly as requested
					float dist = glm::distance(otherPos, boidPos);

					// Rule 2: Separation - avoid crowding neighbors
					if (dist < rule2Distance) {
						rule2Vector -= diff; // Using diff directly
					}

					// Rule 1: Cohesion - fly towards center of mass
					if (dist < rule1Distance) {
						rule1Vector += otherPos;
						rule1Count++;
					}

					// Rule 3: Alignment - match velocity with nearby boids
					if (dist < rule3Distance) {
						rule3Vector += vel1[boidIndex]; // Scattered access
						rule3Count++;
					}
				}
			}
		}
	}

	// Apply Rule 2 (Separation)
	velChange += rule2Vector * rule2Scale;

	// Apply Rule 1 (Cohesion)
	if (rule1Count > 0) {
		rule1Vector /= rule1Count;
		velChange += (rule1Vector - boidPos) * rule1Scale;
	}

	// Apply Rule 3 (Alignment)
	if (rule3Count > 0) {
		rule3Vector /= rule3Count;
		velChange += rule3Vector * rule3Scale;
	}

	// Apply the velocity change
	glm::vec3 newVel = vel1[index] + velChange;

	// Clamp the speed to maximum
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// Store the new velocity
	vel2[index] = newVel;
}

/**
* Update a boid's velocity using the uniform grid with coherent memory access.
* Corresponds to kernUpdateVelNeighborSearchCoherent in the reference.
*/
__global__ void kernUpdateVelNeighborSearchCoherent(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int *gridCellStartIndices, int *gridCellEndIndices,
	glm::vec3 *coherentPos, glm::vec3 *coherentVel, glm::vec3 *vel2) { // vel2 is the output buffer (corresponds to dev_vel2 in reference call)
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) return;

	glm::vec3 boidPos = coherentPos[index]; // Coherent access
	glm::vec3 velChange(0.0f);
	glm::vec3 percieved_center_of_mass(0.0f);
	glm::vec3 separate_vector(0.0f);
	glm::vec3 perceived_velocity(0.0f);
	int neighborCount1 = 0;
	int neighborCount3 = 0;

	glm::ivec3 boidGridPos = (boidPos - gridMin) * inverseCellWidth;
	int gridX = boidGridPos.x;
	int gridY = boidGridPos.y;
	int gridZ = boidGridPos.z;

	// Reordered loops for x→y→z order (with x innermost) to improve memory coalescing
	for (int i = -1; i <= 1; i++) { // x-axis (innermost)
		int neighborX = min(max(gridX + i, 0), gridResolution - 1);

		for (int j = -1; j <= 1; j++) { // y-axis (middle)
			int neighborY = min(max(gridY + j, 0), gridResolution - 1);

			for (int k = -1; k <= 1; k++) { // z-axis (outermost)
				int neighborZ = min(max(gridZ + k, 0), gridResolution - 1);

				int gridIndex = gridIndex3Dto1D(neighborX, neighborY, neighborZ, gridResolution);
				if (gridCellStartIndices[gridIndex] != -1) {
					// Iterate over all boids in this cell (using inclusive end index)
					for (int h = gridCellStartIndices[gridIndex]; h <= gridCellEndIndices[gridIndex]; h++) { // Loop is <= cellEnd
						if (h != index) {
							glm::vec3 otherPos = coherentPos[h]; // Coherent access
							glm::vec3 diff = otherPos - boidPos;
							// Using glm::distance directly as requested
							float dist = glm::distance(otherPos, boidPos);

							// Rule 2: Separation - avoid crowding neighbors
							if (dist < rule2Distance) {
								separate_vector -= diff; // Using diff directly
							}

							// Rule 1: Cohesion - fly towards center of mass
							if (dist < rule1Distance) {
								percieved_center_of_mass += otherPos;
								neighborCount1++;
							}

							// Rule 3: Alignment - match velocity with nearby boids
							if (dist < rule3Distance) {
								perceived_velocity += coherentVel[h]; // Coherent access
								neighborCount3++;
							}
						}
					}
				}
			}
		}
	}

	// Apply Rule 2 (Separation)
	velChange += separate_vector * rule2Scale;

	// Apply Rule 1 (Cohesion)
	if (neighborCount1 > 0) {
		percieved_center_of_mass /= neighborCount1;
		velChange += (percieved_center_of_mass - boidPos) * rule1Scale;
	}

	// Apply Rule 3 (Alignment)
	if (neighborCount3 > 0) {
		perceived_velocity /= neighborCount3;
		velChange += perceived_velocity * rule3Scale;
	}

	// Apply the velocity change
	// Note: The reference adds velChange to coherentVel[index], then clamps
	glm::vec3 currentVel = coherentVel[index]; // Read current velocity from coherent buffer
	glm::vec3 newVel = currentVel + velChange;

	// Clamp the speed to maximum
	float speed = glm::length(newVel);
	if (speed > maxSpeed) {
		newVel = glm::normalize(newVel) * maxSpeed;
	}

	// Store the new velocity in the output buffer (vel2)
	vel2[index] = newVel;
}


/**
* Step the entire N-body simulation by `dt` seconds using the naive brute-force approach.
*/
void Boids::stepSimulationNaive(float dt) {
	//Step the simulation forward in time.
	//Setup thread/block execution configuration
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	//update boid velocities
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(numObjects, d_boidPositions, d_velocitiesA, d_velocitiesB);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	//update boid positions
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(numObjects, dt, d_boidPositions, d_velocitiesB);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	//Ping-pong the velocity buffers
	std::swap(d_velocitiesA, d_velocitiesB);

	// Removed redundant cudaDeviceSynchronize
}

/**
* Step the simulation using the spatial uniform grid optimization with scattered memory access.
*/
void Boids::stepSimulationScatteredGrid(float dt) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid_gridsize((gridCellCount + blockSize - 1) / blockSize); // Use grid size for resetting cell buffers

	// 1) Reset grid cell start/end indices
	kernResetIntBuffer<<<fullBlocksPerGrid_gridsize, blockSize>>>(gridCellCount, d_gridCellStart, -1);
	kernResetIntBuffer<<<fullBlocksPerGrid_gridsize, blockSize>>>(gridCellCount, d_gridCellEnd, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");

	// 2) Compute indices for boid array and grid cells
	kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
		d_boidPositions, d_boidArrayIdx, d_boidGridIdx);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// 3) Sort boids by grid cell index using Thrust
	thrust::device_ptr<int> d_thrust_boidGridIdx(d_boidGridIdx); // Wrap locally
	thrust::device_ptr<int> d_thrust_boidArrayIdx(d_boidArrayIdx); // Wrap locally
	thrust::sort_by_key(thrust::device, d_thrust_boidGridIdx, d_thrust_boidGridIdx + numObjects, d_thrust_boidArrayIdx);
	checkCUDAErrorWithLine("thrust::sort_by_key failed!");

	// 4) Identify start/end indices of each cell in the sorted arrays
	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
		numObjects, d_boidGridIdx, d_gridCellStart, d_gridCellEnd);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// 5) Update velocities using grid-based neighbor search (scattered access)
	kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		d_gridCellStart, d_gridCellEnd, d_boidArrayIdx,
		d_boidPositions, d_velocitiesA, d_velocitiesB);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	// 6) Update positions using the newly calculated velocities
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
		numObjects, dt, d_boidPositions, d_velocitiesB);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// 7) Ping-pong the velocity buffers for the next step
	std::swap(d_velocitiesA, d_velocitiesB);

	// Removed redundant cudaDeviceSynchronize
}

/**
* Step the simulation using the spatial uniform grid optimization with coherent memory access.
* Rewritten to match the structure and buffer flow of the reference implementation.
*/
void Boids::stepSimulationCoherentGrid(float dt)
{
	dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);   // for grid cell operations
	dim3 objBlocks((numObjects + blockSize - 1) / blockSize);       // for per-boid kernels

	// Step 1: Reset grid cell start/end indices - use cellBlocks
	kernResetIntBuffer<<<cellBlocks, blockSize>>>(gridCellCount, d_gridCellStart, -1);
	kernResetIntBuffer<<<cellBlocks, blockSize>>>(gridCellCount, d_gridCellEnd, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer failed!");

	// Step 2: Compute boid indices - use objBlocks
	kernComputeIndices<<<objBlocks, blockSize>>>(
		numObjects, gridSideCount,
		gridMinimum, gridInverseCellWidth,
		d_boidPositions, d_boidArrayIdx, d_boidGridIdx);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// Step 3: Sort boids by grid cell index using Thrust (every frame)
	// Uses d_boidGridIdx (keys) and d_boidArrayIdx (values)
	thrust::device_ptr<int> d_thrust_boidGridIdx_local(d_boidGridIdx); // Wrap locally
	thrust::device_ptr<int> d_thrust_boidArrayIdx_local(d_boidArrayIdx); // Wrap locally
	thrust::sort_by_key(thrust::device, d_thrust_boidGridIdx_local, d_thrust_boidGridIdx_local + numObjects, d_thrust_boidArrayIdx_local);
	checkCUDAErrorWithLine("thrust sorting failed!");

	// Step 4: Identify start/end indices of each cell in the sorted arrays - use objBlocks
	kernIdentifyCellStartEnd<<<objBlocks, blockSize>>>(
		numObjects, d_boidGridIdx, d_gridCellStart, d_gridCellEnd);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// Step 5: Rearrange boid data (positions and velocities) to be coherent in memory - use objBlocks
	kernRearrangeBoidData<<<objBlocks, blockSize>>>(
		numObjects, d_boidArrayIdx,
		d_boidPositions, d_velocitiesA, // Source from main buffers
		d_sortedPositions, d_sortedVelocitiesA); // Destination to coherent buffers
	checkCUDAErrorWithLine("kernRearrangeBoidData failed!");

	// Step 6: Update velocities using coherent grid-based neighbor search - use objBlocks
	kernUpdateVelNeighborSearchCoherent<<<objBlocks, blockSize>>>(
		numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
		d_gridCellStart, d_gridCellEnd,
		d_sortedPositions, d_sortedVelocitiesA, // Inputs from coherent buffers
		d_velocitiesB); // Output to main vel B (temp buffer)
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

	// Step 7: Update positions using the newly calculated velocities - use objBlocks
	kernUpdatePos<<<objBlocks, blockSize>>>(
		numObjects, dt, d_sortedPositions, d_velocitiesB);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// Step 8: Three-step buffer swap as specified in requirements
	// 1) main velocity ping-pong
	std::swap(d_velocitiesA, d_velocitiesB);

	// 2) main position ↔ new coherent positions
	std::swap(d_boidPositions, d_sortedPositions);

	// 3) make next-frame coherent velocity input the latest results
// 3) make next-frame coherent velocity INPUT a **copy** of the new velocities
cudaMemcpy(d_sortedVelocitiesA,            // dst (coherent buffer)
           d_velocitiesA,                  // src (fresh velocities)
           numObjects * sizeof(glm::vec3),
           cudaMemcpyDeviceToDevice);
checkCUDAErrorWithLine("copy vel -> coherent failed!");
}


//Free memory that was allocated in initSimulation
void Boids::endSimulation()
{
	//Free basic boid buffers
	cudaFree(d_boidPositions);
	cudaFree(d_velocitiesA);
	cudaFree(d_velocitiesB);

	//Free grid-related buffers
	cudaFree(d_boidArrayIdx);
	cudaFree(d_boidGridIdx);
	cudaFree(d_gridCellStart);
	cudaFree(d_gridCellEnd);

	//Free coherent grid buffers
	cudaFree(d_sortedPositions);
	cudaFree(d_sortedVelocitiesA);
	// Removed freeing of d_sortedVelocitiesB

	cudaThreadSynchronize(); // Keep final sync in endSimulation
}


void Boids::unitTest()
{
	// Test unstable sort
	int *dev_intKeys;
	int *dev_intValues;
	int N = 12; // Changed from 10 to 12

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;
	intKeys[10] = 4; intValues[10] = 10; // Added two new entries
	intKeys[11] = 1; intValues[11] = 11;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++)
	{
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);
	// Example for using thrust::sort_by_key
	thrust::sort_by_key(thrust::device, dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values); // Added execution policy

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++)
	{
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}

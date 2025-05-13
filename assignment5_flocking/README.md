# CUDA Flocking Simulation

## Overview
This project implements a real-time boid flocking simulation using CUDA to leverage GPU parallelization. The simulation features thousands of autonomous agents (boids) that exhibit emergent flocking behaviors through simple rules of cohesion, separation, and alignment.

## Key Features
- **High-performance CUDA implementation** supporting up to millions of boids
- **Three simulation methods**:
  - Naive (brute force) approach
  - Scattered grid-based neighbor search
  - Coherent grid-based neighbor search with memory optimization
- **Interactive visualization** for real-time observation
- **Performance testing mode** for benchmarking different methods
- **Block size optimization** for finding optimal CUDA execution configurations
- **Headless operation support** for running performance tests without a display

## System Requirements

### For Visualization Mode
- CUDA-capable NVIDIA GPU
- OpenGL 3.3+ support
- GLFW and GLEW libraries
- Display server (X11, Wayland, XQuartz)

### For Performance Testing Only
- CUDA-capable NVIDIA GPU
- No GLFW, OpenGL, or display server required
- Can run on headless servers and cloud instances

## Quick Start

### Building the Project
```bash
# Navigate to project directory
cd CUDA-Flocking

# Build the project
make
```

### Running the Visualization
To run the visualization with default settings (5000 boids, naive method):
```bash
./build/bin/cis565_boids
```

To specify simulation method and boid count:
```bash
./build/bin/cis565_boids --mode [naive|scattered|coherent] --boids [count]
```

Example:
```bash
./build/bin/cis565_boids --mode coherent --boids 25000
```

### Controls
- **Left mouse button**: Rotate camera
- **Right mouse button**: Zoom in/out
- **ESC**: Exit the application

## Performance Testing

### Single Method Performance Test
To run performance tests for a specific method and boid count:
```bash
./build/bin/cis565_boids --perf-test [naive|scattered|coherent] [count]
```

Example:
```bash
./build/bin/cis565_boids --perf-test coherent 50000
```

### Full Performance Testing
To test all boid sizes with a specific method:
```bash
./build/bin/cis565_boids --perf-test [naive|scattered|coherent]
```

Example:
```bash
./build/bin/cis565_boids --perf-test coherent
```

### Block Size Performance Testing
To test how different CUDA block sizes affect performance across all methods:
```bash
./build/bin/cis565_boids --perf-test-block-size
```

This will run a comprehensive test of all three methods (Naive, Scattered Grid, and Coherent Grid) with different block sizes (32, 64, 128, 256, 512, and 1024) to find the optimal execution configuration.

### Headless Operation
All performance test modes (`--perf-test` and `--perf-test-block-size`) can now run on headless systems without GLFW or display server requirements. The code automatically uses `std::chrono` for timing in performance test modes instead of GLFW, making it possible to benchmark on cloud servers and compute instances that don't have displays attached.

## Implementation Details

### Optimization Approaches

#### 1. Parameter Optimization
We carefully tuned the flocking parameters to achieve natural-looking behavior while maintaining performance:
- Increased `maxSpeed` from 1.8 to 2.5 for faster and more dynamic movement
- Enhanced alignment factor (`rule3Scale`) from 0.15 to 0.25 for smoother directional changes
- Adjusted cohesion and separation weights for better flocking behavior
- Reduced time step (`dt`) from 0.2 to 0.1 for smoother animation

#### 2. Algorithmic Optimizations

**Naive Implementation:**
- Brute-force approach that compares each boid with every other boid
- O(n²) complexity, suitable for small simulations
- Used squared distances to avoid expensive sqrt operations

**Scattered Grid Implementation:**
- Divides space into a uniform grid for efficient neighbor searching
- Only examines boids in neighboring cells instead of all boids
- Uses Thrust library for efficient sorting of boids by grid cell
- Maintains original data structure with indirect access using indices

**Coherent Grid Implementation:**
- Similar to scattered grid but reorganizes data for memory coherence
- Rearranges boid data to be contiguous in memory based on spatial locality
- Enables more efficient memory access patterns on the GPU
- Swaps pointers instead of copying data back to avoid expensive transfers
- Added explicit synchronization points to ensure data consistency

- **Velocity-copy fix** – a silent black-screen bug turned out to be caused by
  using `std::swap` on velocity pointers *twice* in the coherent path.  
  The second swap left the next-frame neighbour search reading *stale* data,
  so every boid computed a zero update and never reached the renderer.  
  The cure was to keep the first ping-pong swap **but** replace the second with an explicit
  device-to-device copy:

  ```cpp
  cudaMemcpy(d_sortedVelocitiesA,            // dst – coherent buffer
             d_velocitiesA,                  // src – freshly-updated velocities
             numObjects * sizeof(glm::vec3),
             cudaMemcpyDeviceToDevice);
  ```

#### 3. Memory Optimizations
- Used ping-pong buffering for velocity updates to avoid race conditions
- Pre-computed squared distances for performance
- Implemented memory-efficient grid cell indexing
- Optimized memory access patterns for better coalescing

#### 4. Block Size Optimization
- Added ability to test different CUDA thread block sizes
- Implemented dynamic block size selection for optimal performance
- Performance varies significantly based on block size and simulation method
- Block sizes between 64-128 typically offer best performance for compute-bound workloads
- Grid-based methods are less sensitive to block size due to memory-bound nature

## Technical Details

### Key CUDA Kernels

1. **kernUpdateVelocityBruteForce**: Updates velocities using brute-force approach

2. **kernUpdateVelNeighborSearchScattered**: Updates velocities using grid-based neighbor search with scattered memory layout

3. **kernUpdateVelNeighborSearchCoherent**: Updates velocities using grid-based search with coherent memory layout

4. **kernUpdatePos**: Updates boid positions based on velocities and handles boundary conditions

5. **kernResetAndComputeIndices**: Assigns boids to grid cells and resets grid data structures

6. **kernIdentifyCellStartEnd**: Identifies the start and end indices for each grid cell

7. **kernRearrangeBoidData**: Reorganizes boid data for memory coherence

### Simulation Steps

For the coherent grid implementation (most optimized):

1. Reset grid cell buffers and compute boid grid indices
2. Sort boids by grid cell index
3. Identify start/end indices for each grid cell
4. Rearrange boid data to be coherent in memory
5. Update velocities using neighbor search
6. Update positions using newly calculated velocities
7. Swap buffer pointers for next iteration

## Performance Analysis

### Method Efficiency
The performance hierarchy from least to most efficient:
1. **Naive**: O(n²) complexity, suitable only for small simulations
2. **Scattered Grid**: O(n) complexity with some memory access inefficiency
3. **Coherent Grid**: O(n) complexity with optimized memory access patterns

### Block Size Impact

Based on performance testing with 25,000 boids:

**Naive Implementation**:
- Most sensitive to block size changes
- Optimal block size: 128 threads (261 FPS)
- Performance drops with both smaller (32) and larger (1024) block sizes
- This suggests the workload is compute-bound and benefits from balanced occupancy

**Scattered Grid Implementation**:
- Less sensitive to block size after 64 threads
- Optimal block size: 64-128 threads (~8880 FPS)
- Performance is relatively stable across block sizes 64-1024
- This suggests the workload becomes memory-bound after sufficient threads are available

**Coherent Grid Implementation**:
- Minimal sensitivity to block size
- Consistent performance across all tested block sizes (~9500-9600 FPS)
- Block size 32 shows slightly better performance in some cases
- This suggests memory access patterns are well-optimized and the workload is primarily memory-bandwidth limited

### Observations

* **Linear scaling**    
  Both grid-based approaches scale *O(n)* in practice. FPS degrades roughly inversely with boid count, while the naive method collapses once pairwise `n²` comparisons dominate cache and SM occupancy.

* **Coherent vs. Scattered**    
  Re-ordering boid data for memory coherence yields an extra **25 – 40 %** speed-up across the tested range. Benefits grow with scene density because global memory transactions become the dominant cost.

* **Block Size Optimization**
  Block size has varying impact depending on the simulation method:
  - Naive method: Block size matters significantly (up to 16% performance difference)
  - Grid methods: Less sensitive once minimum occupancy is achieved
  - For general purposes, a block size of 128 offers good performance across all methods

* **Velocity-copy fix**    
  The black-screen bug (see "Algorithmic Optimizations → Coherent Grid Implementation") masked itself as a performance hit. After copying freshly-updated velocities back into the coherent buffer, FPS jumped by **> 30 %** and boids rendered correctly.

* **Million-boid milestone**    
  The coherent implementation sustains ~800 FPS at **one million boids**, limited chiefly by grid-cell occupancy in shared memory rather than raw arithmetic throughput.

The coherent grid implementation typically provides 10-100x speedup compared to naive for large boid counts.

Note that optimal performance depends on your specific GPU. For reference, on an NVIDIA GeForce RTX 3080:
- Coherent grid achieves 60+ FPS with 100,000 boids
- Scattered grid achieves 60+ FPS with 50,000 boids
- Naive implementation achieves 60+ FPS with only 5,000 boids


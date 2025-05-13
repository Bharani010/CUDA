#include "common.h"
#include <cuda.h>

#define TPB 256  // Threads per block for CUDA kernels

// Constants used by all kernels, copied to GPU memory once at startup
__constant__ double c_cutoff, c_min_r2, c_dt;
__constant__ int    c_ncells_row;

// Helper function to find which cell a particle belongs to based on its position
__device__ __forceinline__ int cell_of(double x, double y)
{
    int cx = (int)(x / c_cutoff);
    int cy = (int)(y / c_cutoff);
    cx = max(0, min(cx, c_ncells_row - 1));
    cy = max(0, min(cy, c_ncells_row - 1));
    return cx + cy * c_ncells_row;
}

// Helper function to calculate the force between two particles
__device__ void apply_force(particle_t &p, const particle_t &q)
{
    double dx = q.x - p.x;
    double dy = q.y - p.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > c_cutoff * c_cutoff || r2 == 0.0) return;   // Skip if too far or same particle
    r2 = (r2 > c_min_r2) ? r2 : c_min_r2;
    double r    = sqrt(r2);
    double coef = (1.0 - c_cutoff / r) / r2 / mass;
    p.ax += coef * dx;
    p.ay += coef * dy;
}

// Kernel to build linked lists of particles in each cell
__global__ void build_list(const particle_t *parts, int *head, int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int cid  = cell_of(parts[tid].x, parts[tid].y);
    next[tid] = atomicExch(head + cid, tid);   // Add particle to the front of its cell's list
}

// Kernel to compute forces for each particle using nearby cells
__global__ void compute_forces(particle_t *parts, const int *head, const int *next, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    particle_t p = parts[tid];
    p.ax = p.ay = 0.0;  // Reset this particle’s acceleration

    int cx = (int)(p.x / c_cutoff);
    int cy = (int)(p.y / c_cutoff);

    // Check this particle’s cell and its 8 neighbors
    for (int dy = -1; dy <= 1; ++dy) {
        int ny = cy + dy; if (ny < 0 || ny >= c_ncells_row) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = cx + dx; if (nx < 0 || nx >= c_ncells_row) continue;
            for (int q = head[nx + ny * c_ncells_row]; q != -1; q = next[q])
                apply_force(p, parts[q]);
        }
    }

    parts[tid].ax = p.ax;
    parts[tid].ay = p.ay;
}

// Kernel to update particle positions and velocities after forces are calculated
__global__ void move_particles(particle_t *parts, int n, double box_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    particle_t p = parts[tid];

    p.vx += p.ax * c_dt;
    p.vy += p.ay * c_dt;
    p.x  += p.vx * c_dt;
    p.y  += p.vy * c_dt;

    // Bounce particles off the walls if they go outside the box
    while (p.x < 0 || p.x > box_size) {
        p.x  = (p.x < 0) ? -p.x : 2 * box_size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > box_size) {
        p.y  = (p.y < 0) ? -p.y : 2 * box_size - p.y;
        p.vy = -p.vy;
    }

    parts[tid] = p;
}

// Variables on the GPU for tracking cells and particle links
static int *d_head = nullptr, *d_next = nullptr;
static int  ncells_row, ncells;

// Set up the simulation by copying constants and allocating GPU memory
void init_simulation(particle_t *parts, int n, double size)
{
    double cutoff_h   = cutoff;
    double min_r2_h   = min_r * min_r;
    double dt_h       = dt;
    ncells_row        = (int)(size / cutoff) + 1;
    ncells            = ncells_row * ncells_row;

    cudaMemcpyToSymbol(c_cutoff,     &cutoff_h,  sizeof(double));
    cudaMemcpyToSymbol(c_min_r2,     &min_r2_h,  sizeof(double));
    cudaMemcpyToSymbol(c_dt,         &dt_h,      sizeof(double));
    cudaMemcpyToSymbol(c_ncells_row, &ncells_row,sizeof(int));

    cudaMalloc(&d_head, ncells * sizeof(int));
    cudaMalloc(&d_next, n       * sizeof(int));
}

// Run one step of the simulation: build lists, compute forces, move particles
void simulate_one_step(particle_t *parts, int n, double size)
{
    int blks = (n + TPB - 1) / TPB;

    cudaMemset(d_head, 0xff, ncells * sizeof(int));  // Clear cell lists by setting to -1
    build_list     <<<blks, TPB>>>(parts, d_head, d_next, n);
    compute_forces <<<blks, TPB>>>(parts, d_head, d_next, n);
    move_particles <<<blks, TPB>>>(parts, n, size);
}
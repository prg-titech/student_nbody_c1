#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <new>

#include <curand_kernel.h>

#include "cuda_helper.h"
#include "nbody.h"
#include "rendering.h"

// Simulation parameters.
static const int kSeed = 42;
static const float kTimeInterval = 0.5;
static const int kBenchmarkIterations = 10000;

// Physical constants.
static const float kGravityConstant = 6.673e-11;   // gravitational constant

// Array containing all Body objects on device.
Body* host_bodies;
__device__ Body* dev_bodies;


__device__ Body::Body(float pos_x, float pos_y,
                      float vel_x, float vel_y, float mass) {
                        pos_x_ = pos_x;
                        pos_y_ = pos_y;
                        vel_x_ = vel_x;
                        vel_y_ = vel_y;
                        mass_ = mass;
                        force_x_ = 0;
                        force_y_ = 0;
}


__device__ void Body::compute_force() {
  force_x_ = 0;
  force_y_ = 0;
  for (int i = 0; i < kNumBodies; ++i){
    if(this != (dev_bodies + i)){
      float m1 = dev_bodies[i].mass_;
      float dx = dev_bodies[i].pos_x_ - pos_x_;
      float dy = dev_bodies[i].pos_y_ - pos_y_;
      float r = sqrt(dx * dx + dy * dy);
      force_x_ += kGravityConstant * m1 * mass_ / (r * r * r) * dx;
      force_y_ += kGravityConstant * m1 * mass_ / (r * r * r) * dy;
    }
  }
}


__device__ void Body::update(float dt) {
  vel_x_ += force_x_ / mass_ * dt;
  vel_y_ += force_y_ / mass_ * dt;
  pos_x_ += vel_x_ * dt;
  pos_y_ += vel_y_ * dt;
  if (abs(pos_x_) > 1) {
    vel_x_ = - vel_x_;
  }
  if (abs(pos_y_) > 1) {
    vel_y_ = - vel_y_;
  }
  // Bodies should bounce off the wall when they go out of range.
  // Range: [-1, -1] to [1, 1]
}


__global__ void kernel_initialize_bodies(Body* bodies) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    dev_bodies = bodies;

    // Initialize random state.
    curandState rand_state;
    curand_init(kSeed, i, 0, &rand_state);

    // Create new Body object with placement-new.
    new(dev_bodies + i) Body(/*pos_x=*/ 2 * curand_uniform(&rand_state) - 1,
                             /*pos_y=*/ 2 * curand_uniform(&rand_state) - 1,
                             /*vel_x=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
                             /*vel_y=*/ (curand_uniform(&rand_state) - 0.5) / 1000,
                             /*mass=*/ (curand_uniform(&rand_state)/2 + 0.5)
                                           * kMaxMass);
  }
}


__global__ void kernel_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    dev_bodies[i].compute_force();
  }
}


__global__ void kernel_update() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
       i < kNumBodies; i += blockDim.x * gridDim.x) {
    dev_bodies[i].update(kTimeInterval);
  }
}


// Compute one step of the simulation.
void step_simulation() {
  // n-body consists of 2 CUDA kernels.
  // The first kernel computes the total accumulated gravitational force for
  // every body. The second kernel updates every body's velocity and position.

  kernel_compute_force<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_update<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());
}



void run_interactive() {
  init_renderer();

  // Run simulation until user closes the window.
  do {
    // Compute one step.
    step_simulation();
  } while (draw(host_bodies));

  close_renderer();
}

void run_benchmark() {
  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kBenchmarkIterations; ++i) {
    step_simulation();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
      .count();

  printf("Time: %lu ms\n", millis);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s mode\n\nmode 0: Interactive mode\nmode 1: Benchmark\n",
           argv[0]);
    return 1;
  }

  int mode = atoi(argv[1]);

  // Allocate and create Body objects.
  cudaMallocManaged(&host_bodies, sizeof(Body)*kNumBodies);
  kernel_initialize_bodies<<<128, 128>>>(host_bodies);
  gpuErrchk(cudaDeviceSynchronize());

  if (mode == 0) {
    run_interactive();
  } else if (mode == 1) {
    run_benchmark();
  } else {
    printf("Invalid mode.\n");
    return 1;
  }

  cudaFree(host_bodies);
  return 0;
}

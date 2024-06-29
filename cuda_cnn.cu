#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Define constants
#define IMAGE_SIZE 28
#define KERNEL_SIZE 3
#define NUM_FILTERS 32
#define POOL_SIZE 2
#define FC_SIZE 128
#define NUM_CLASSES 10

// Convolutional layer kernel
__global__ void convLayerKernel(float* input, float* filters, float* output, int inputSize, int numFilters) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x < inputSize - KERNEL_SIZE + 1 && y < inputSize - KERNEL_SIZE + 1 && f < numFilters) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                sum += input[(y + i) * inputSize + (x + j)] * filters[f * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j];
            }
        }
        output[f * (inputSize - KERNEL_SIZE + 1) * (inputSize - KERNEL_SIZE + 1) + y * (inputSize - KERNEL_SIZE + 1) + x] = sum;
    }
}

// Max pooling layer kernel
__global__ void maxPoolKernel(float* input, float* output, int inputSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (x < inputSize / POOL_SIZE && y < inputSize / POOL_SIZE) {
        float maxVal = -INFINITY;
        for (int i = 0; i < POOL_SIZE; i++) {
            for (int j = 0; j < POOL_SIZE; j++) {
                float val = input[f * inputSize * inputSize + (y * POOL_SIZE + i) * inputSize + (x * POOL_SIZE + j)];
                maxVal = fmaxf(maxVal, val);
            }
        }
        output[f * (inputSize / POOL_SIZE) * (inputSize / POOL_SIZE) + y * (inputSize / POOL_SIZE) + x] = maxVal;
    }
}

// ReLU activation function kernel
__global__ void reluKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Fully connected layer kernel
__global__ void fcLayerKernel(float* input, float* weights, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = sum;
    }
}

// Softmax function kernel
__global__ void softmaxKernel(float* input, float* output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += expf(input[i]);
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]) / sum;
    }
}

int main() {
    // Allocate memory and initialize data (not shown for brevity)
    // ...

    // Define grid and block dimensions
    dim3 convBlockDim(16, 16);
    dim3 convGridDim((IMAGE_SIZE - KERNEL_SIZE + 1 + 15) / 16, (IMAGE_SIZE - KERNEL_SIZE + 1 + 15) / 16, NUM_FILTERS);

    dim3 poolBlockDim(16, 16);
    dim3 poolGridDim(((IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE + 15) / 16, ((IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE + 15) / 16, NUM_FILTERS);

    // Launch kernels
    convLayerKernel<<<convGridDim, convBlockDim>>>(d_input, d_filters, d_conv_output, IMAGE_SIZE, NUM_FILTERS);
    reluKernel<<<(NUM_FILTERS * (IMAGE_SIZE - KERNEL_SIZE + 1) * (IMAGE_SIZE - KERNEL_SIZE + 1) + 255) / 256, 256>>>(d_conv_output, NUM_FILTERS * (IMAGE_SIZE - KERNEL_SIZE + 1) * (IMAGE_SIZE - KERNEL_SIZE + 1));
    maxPoolKernel<<<poolGridDim, poolBlockDim>>>(d_conv_output, d_pool_output, IMAGE_SIZE - KERNEL_SIZE + 1);
    
    fcLayerKernel<<<(FC_SIZE + 255) / 256, 256>>>(d_pool_output, d_fc_weights, d_fc_output, NUM_FILTERS * ((IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE) * ((IMAGE_SIZE - KERNEL_SIZE + 1) / POOL_SIZE), FC_SIZE);
    reluKernel<<<(FC_SIZE + 255) / 256, 256>>>(d_fc_output, FC_SIZE);
    
    fcLayerKernel<<<(NUM_CLASSES + 255) / 256, 256>>>(d_fc_output, d_output_weights, d_output, FC_SIZE, NUM_CLASSES);
    softmaxKernel<<<1, NUM_CLASSES>>>(d_output, d_probabilities, NUM_CLASSES);

    // Copy results back to host and free memory (not shown for brevity)
    // ...

    return 0;
}
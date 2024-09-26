/**
 * @file       matrix_mul.cu
 * @author     Pinaki Nath Chowdhury
 * @brief      Uses cuBLAS routine `cublasSgemm` to multiply two matrices
 *             For storing the matrices in Host, we use Heap memory
 *             instead of stack memory used in `matrix_mul_stack.cu`.
 *             It is important to learn to play with heap memory with safety checks.
 * @copyright  Do whatever you want. Don't ask me.
 */

#include <stdio.h>

// GPU / CUDA headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]) {
    const int rowsA = 2, colsA = 3;
    const int rowsB = 3, colsB = 2;

    // Allocate pinned memory (i.e., page-locked) on the heap
    float *h_A, *h_B, *h_C;
    cudaError_t err_hA = cudaMallocHost((void **)&h_A, rowsA * colsA * sizeof(float));
    cudaError_t err_hB = cudaMallocHost((void **)&h_B, rowsB * colsB * sizeof(float));
    cudaError_t err_hC = cudaMallocHost((void **)&h_C, rowsA * colsB * sizeof(float));

    if (err_hA != cudaSuccess || err_hB != cudaSuccess || err_hC != cudaSuccess) {
        printf("Failed to allocate Page-locked heap memory.\n");
        printf("Error Status of h_A: %s\n", cudaGetErrorString(err_hA));
        printf("Error Status of h_B: %s\n", cudaGetErrorString(err_hB));
        printf("Error Status of h_C: %s\n", cudaGetErrorString(err_hC));
        
        return EXIT_FAILURE;
    }
    
    // Assign values to Array
    for (int i = 0; i < rowsA * colsA; ++i) {
        h_A[i] = (float)i+1;
    }
    
    for (int i = 0; i < rowsB * colsB; ++i) {
        h_B[i] = (float)i+1;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, rowsA * colsA * sizeof(float));
    cudaMalloc((void **)&d_B, rowsB * colsB * sizeof(float));
    cudaMalloc((void **)&d_C, rowsA * colsB * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication C = A * B
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        rowsA, colsB, colsA, &alpha,
        d_A, rowsA,
        d_B, rowsB,
        &beta,
        d_C, rowsA);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Matrix C:\n");
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            printf("%.4f \t", h_C[(rowsA * i) + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return EXIT_SUCCESS;
}
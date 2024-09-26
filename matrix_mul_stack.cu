/**
 * @file       matrix_mul.cu
 * @author     Pinaki Nath Chowdhury
 * @brief      Uses cuBLAS routine `cublasSgemm` to multiply two matrices
 *             For storing the matrices in Host, we use Stack memory.
 *             However, in many cases, you won't be able to use stack memory.
 * @copyright  Do whatever you want. Don't ask me.
 */

#include <stdio.h>

// GPU / CUDA headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char *argv[]) {
    const int rowsA = 2, colsA = 3;
    const int rowsB = 3, colsB = 2;

    // Allocate stack host memory
    float h_A[rowsA * colsA] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_B[rowsB * colsB] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_C[rowsA * colsB]  = {0.0f, 0.0f, 0.0f, 0.0f};

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

    return EXIT_SUCCESS;
}
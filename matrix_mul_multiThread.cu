/**
 * @file      matrix_mul_multiThread.cu
 * @author    Pinaki Nath Chowdhury
 * @brief     Uses cuBLAS routine `cublasSgemm` to multiply two matrices
 *            We demonstrate Multi-Threaded cuBLAS Matrix Multiplication.
 * @copyright Do whatever you want. Don't ask me.
 */

#include <stdio.h>
#include <pthread.h>

// GPU / CUDA headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct {
    float *A, *B, *C;
    int rowsA, colsA, rowsB, colsB;
    cublasHandle_t handle;
} MatrixMulData;

void *matrixMultiply(void *args) {
    MatrixMulData *data = (MatrixMulData *)args;

    // Allocate GPU memory for matrices
    float *d_A, *d_B, *d_C;
    cudaError_t err_dA = cudaMalloc((void **)&d_A, data->rowsA * data->colsA * sizeof(float));
    cudaError_t err_dB = cudaMalloc((void **)&d_B, data->rowsB * data->colsB * sizeof(float));
    cudaError_t err_dC = cudaMalloc((void **)&d_C, data->rowsA * data->colsB * sizeof(float));

    // Check for errors when allocating device memory
    if (err_dA != cudaSuccess || err_dB != cudaSuccess || err_dC != cudaSuccess) {
        printf("Failed to allocate Device Memory.\n");
        printf("Error status A: %s\n", cudaGetErrorString(err_dA));
        printf("Error status B: %s\n", cudaGetErrorString(err_dB));
        printf("Error status C: %s\n", cudaGetErrorString(err_dC));

        return nullptr;
    }

    // Copy matrices from host to device
    cudaMemcpy(d_A, data->A, data->rowsA * data->colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, data->B, data->rowsB * data->colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication C = A * B
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(data->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        data->rowsA, data->colsB, data->colsA, &alpha,
        d_A, data->rowsA,
        d_B, data->rowsB,
        &beta,
        d_C, data->rowsA);
    
    // Copy the result back to the host
    cudaMemcpy(data->C, d_C, data->rowsA * data->colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return nullptr;
}

int main(int argc, char *argv[]) {
    const int rowsA = 2, colsA = 3;
    const int rowsB = 3, colsB = 2;

    // Define matrices for two threads
    float *h_A1, *h_B1, *h_C1;
    float *h_A2, *h_B2, *h_C2;

    // Allocate pinned memory (i.e., page-locked) from the heap
    cudaError_t err_hA1 = cudaMallocHost((void **)&h_A1, rowsA * colsA * sizeof(float));
    cudaError_t err_hB1 = cudaMallocHost((void **)&h_B1, rowsB * colsB * sizeof(float));
    cudaError_t err_hC1 = cudaMallocHost((void **)&h_C1, rowsA * colsB * sizeof(float));

    cudaError_t err_hA2 = cudaMallocHost((void **)&h_A2, rowsA * colsA * sizeof(float));
    cudaError_t err_hB2 = cudaMallocHost((void **)&h_B2, rowsB * colsB * sizeof(float));
    cudaError_t err_hC2 = cudaMallocHost((void **)&h_C2, rowsA * colsB * sizeof(float));

    // Check for failure in allocating memory from heap
    if (err_hA1 != cudaSuccess || err_hB1 != cudaSuccess || err_hC1 != cudaSuccess || err_hA2 != cudaSuccess || err_hB2 != cudaSuccess || err_hC2 != cudaSuccess) {
        printf("Failed to allocate Page-locked heap memory.\n");
        printf("Error Status of h_A1: %s\n", cudaGetErrorString(err_hA1));
        printf("Error Status of h_B1: %s\n", cudaGetErrorString(err_hB1));
        printf("Error Status of h_C1: %s\n", cudaGetErrorString(err_hC1));

        printf("Error Status of h_A2: %s\n", cudaGetErrorString(err_hA2));
        printf("Error Status of h_B2: %s\n", cudaGetErrorString(err_hB2));
        printf("Error Status of h_C2: %s\n", cudaGetErrorString(err_hC2));

        return EXIT_FAILURE;
    }

    // Initialise values to arrays
    for (int i = 0; i < rowsA * colsA; ++i) {
        h_A1[i] = (float) i+1;
        h_A2[i] = (float) 2*(i+1);
    }
    for (int i = 0; i < rowsB * colsB; ++i) {
        h_B1[i] = (float) i+1;
        h_B2[i] = (float) 2*(i+1);
    }
    for (int i = 0; i < rowsA * colsB; ++i) {
        h_C1[i] = 0.0f;
        h_C2[i] = 0.0f;
    }

    // Create cuBLAS handles for each thread
    cublasHandle_t handle1, handle2;
    cublasCreate(&handle1);
    cublasCreate(&handle2);

    // Prepare data for each thread
    MatrixMulData data1 = {h_A1, h_B1, h_C1, rowsA, colsA, rowsB, colsB, handle1};
    MatrixMulData data2 = {h_A2, h_B2, h_C2, rowsA, colsA, rowsB, colsB, handle2};

    // Create threads
    pthread_t thread1, thread2;
    pthread_create(&thread1, nullptr, matrixMultiply, &data1);
    pthread_create(&thread2, nullptr, matrixMultiply, &data2);

    // Join threads
    pthread_join(thread1, nullptr);
    pthread_join(thread2, nullptr);

    // Destroy cuBLAS handles
    cublasDestroy(handle1);
    cublasDestroy(handle2);

    // Print the results
    printf("Result of A1 * B1:\n");
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            printf("%.4f    ", h_C1[(i * rowsA) + j]);
        }
        printf("\n");
    }

    printf("Result of A2 * B2:\n");
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            printf("%.4f    ", h_C2[(i * rowsA) + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFreeHost(h_A1);
    cudaFreeHost(h_B1);
    cudaFreeHost(h_C1);

    cudaFreeHost(h_A2);
    cudaFreeHost(h_B2);
    cudaFreeHost(h_C2);

    return EXIT_SUCCESS;
}
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_fp16.h>

// Utility function to fill matrix with random values and convert to __half
void fillMatrixHalf(__half* matrix, int numRows, int numCols) {
    float* temp = (float*)malloc(numRows * numCols * sizeof(float));
    for (int i = 0; i < numRows * numCols; i++) {
        temp[i] = (float)rand() / RAND_MAX;
    }
    cudaMemcpy(matrix, temp, numRows * numCols * sizeof(__half), cudaMemcpyHostToDevice);
    free(temp);
}

void gemv(__half* A, __half* x, __half* y, int m, int n) {
    // Perform y = A * x using gemv
    cublasHandle_t handle;
    cublasCreate(&handle);
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasHgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1);
    cublasDestroy(handle);
}

// 主函数
int main() {
    srand(time(NULL));

    cublasHandle_t handle;
    cublasCreate(&handle);

    const int numTests = 100; // 测试次数
    const int warmupRuns = 10; // 预热次数

    int seqlen = 1;
    int in_dimension = 4096;
    int out_dimension = 4096;

    // Evaluate direct Y = WX
    __half *W, *X, *Y;
    cudaMalloc(&W, out_dimension * in_dimension * sizeof(__half));
    cudaMalloc(&X, in_dimension * seqlen * sizeof(__half));
    cudaMalloc(&Y, seqlen * out_dimension * sizeof(__half));

    fillMatrixHalf(W, out_dimension, in_dimension);
    fillMatrixHalf(X, in_dimension, seqlen);

    // 预热
    for (int i = 0; i < warmupRuns; ++i) {
        gemv(W, X, Y, out_dimension, in_dimension);
    }

    // 实际测试
    cudaDeviceSynchronize();
    clock_t start = clock();
    for (int i = 0; i < numTests; ++i) {
        gemv(W, X, Y, out_dimension, in_dimension);
    }
    cudaDeviceSynchronize();
    clock_t end = clock();

    // 计算平均执行时间
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC / numTests;
    printf("Average time for direct: %f seconds\n", time_taken);

    for (float param_ratio = 0.75; param_ratio < 1; param_ratio += 0.05) {
        int rank = (int)((param_ratio * in_dimension * out_dimension) / (out_dimension + in_dimension));
        printf("rank: %d\n", rank);

        __half *A, *B, *X, *BX, *Y;
        cudaMalloc(&A, rank * out_dimension * sizeof(__half));
        cudaMalloc(&B, in_dimension * rank * sizeof(__half));
        cudaMalloc(&X, in_dimension * seqlen * sizeof(__half));
        cudaMalloc(&BX, rank * seqlen * sizeof(__half));
        cudaMalloc(&Y, seqlen * out_dimension * sizeof(__half));

        fillMatrixHalf(A, rank, out_dimension);
        fillMatrixHalf(B, in_dimension, rank);
        fillMatrixHalf(X, in_dimension, seqlen);

        // 预热
        for (int i = 0; i < warmupRuns; ++i) {
            gemv(B, X, BX, rank, in_dimension);
            gemv(A, BX, Y, out_dimension, rank);
        }

        // 实际测试
        clock_t start = clock();
        cudaDeviceSynchronize();
        for (int i = 0; i < numTests; ++i) {
            gemv(B, X, BX, rank, in_dimension);
            gemv(A, BX, Y, out_dimension, rank);
        }
        cudaDeviceSynchronize();
        clock_t end = clock();

        // 计算平均执行时间
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC / numTests;
        printf("Average time for param_ratio %f: %f seconds\n", param_ratio, time_taken);

        cudaFree(A);
        cudaFree(B);
        cudaFree(X);
        cudaFree(BX);
        cudaFree(Y);
    }

    cudaFree(W);
    cudaFree(X);
    cudaFree(Y);
    cublasDestroy(handle);
    return 0;
}

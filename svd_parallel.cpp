#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// Function to perform matrix multiplication on GPU
void cuda_matrix_multiply(const MatrixXd& A, const MatrixXd& B, MatrixXd& C) {
    int n = A.rows();
    int m = A.cols();
    int p = B.cols();

    // Allocate memory on the device
    double* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * m * sizeof(double));
    cudaMalloc(&d_B, m * p * sizeof(double));
    cudaMalloc(&d_C, n * p * sizeof(double));

    // Copy data to the device
    cudaMemcpy(d_A, A.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), m * p * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((p + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    // Launch the kernel
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, n, m, p);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, n * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Main function
int main() {
    // Set random seed for reproducibility
    srand(time(0));

    // Define dimensions of the matrix
    int n = 1000; // Number of rows
    int m = 1000; // Number of columns

    // Generate random matrix X with values from a standard normal distribution
    MatrixXd X(n, m);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    auto start_time = chrono::high_resolution_clock::now(); // Start timing

    // Parallelize matrix initialization using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X(i, j) = i + j + dist(gen) * 0.001;
        }
    }

    // Compute mean across columns using OpenMP
    RowVectorXd X_mean(m);
    #pragma omp parallel for reduction(+:X_mean)
    for (int j = 0; j < m; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += X(i, j);
        }
        X_mean(j) = sum / n;
    }

    // Subtract mean to center the data
    MatrixXd B = X;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        B.row(i) -= X_mean;
    }

    // Compute variance of the data using OpenMP
    double var = 0.0;
    #pragma omp parallel for reduction(+:var)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            var += B(i, j) * B(i, j);
        }
    }
    var /= (n * m);

    // Normalize the data by dividing by the square root of variance
    B /= sqrt(var);

    // Compute covariance matrix using CUDA
    MatrixXd C(n, n);
    cuda_matrix_multiply(B.transpose(), B, C);
    C /= (n - 1);

    // Perform eigen decomposition using Eigen library
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(C);
    VectorXd eigenvalues = eigen_solver.eigenvalues(); // Eigenvalues
    MatrixXd eigenvectors = eigen_solver.eigenvectors(); // Eigenvectors

    // Project X onto PC space
    MatrixXd T_pca = B * eigenvectors;

    // Perform Singular Value Decomposition (SVD)
    JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    VectorXd singular_values = svd.singularValues();
    MatrixXd Vt = svd.matrixV().transpose();

    // Construct the Sigma matrix
    MatrixXd Sigma(n, m);
    Sigma.setZero();
    for (int i = 0; i < singular_values.size(); ++i) {
        Sigma(i, i) = singular_values(i);
    }

    // Compute T_svd = U * Sigma
    MatrixXd T_svd = U * Sigma;

    auto end_time = chrono::high_resolution_clock::now(); // End timing
    chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Save results to files
    ofstream pca_file("pca_results_parallel.txt");
    ofstream svd_file("svd_results_parallel.txt");
    ofstream singular_values_file("singular_values_parallel.txt");
    ofstream eigenvalues_file("eigenvalues_parallel.txt");

    if (pca_file.is_open()) {
        pca_file << "Principal components using PCA T_pca = B * V:\n";
        pca_file << T_pca << endl;
        pca_file.close();
    }

    if (svd_file.is_open()) {
        svd_file << "Principal components using SVD T_svd = U * Sigma:\n";
        svd_file << T_svd << endl;
        svd_file.close();
    }

    if (singular_values_file.is_open()) {
        singular_values_file << "Singular values of B matrix:\n";
        singular_values_file << singular_values << endl;
        singular_values_file.close();
    }

    if (eigenvalues_file.is_open()) {
        eigenvalues_file << "Eigenvalues of C matrix:\n";
        eigenvalues_file << eigenvalues << endl;
        eigenvalues_file.close();
    }

    // Print execution time
    cout << "Execution time: " << elapsed_seconds.count() << " seconds" << endl;

    return 0;
}

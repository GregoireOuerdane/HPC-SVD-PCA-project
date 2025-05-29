#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace Eigen;

// Function to compute covariance matrix in parallel using MPI
MatrixXd mpi_covariance(const MatrixXd& B, int n, int m) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Split the rows of B among processes
    int rows_per_proc = n / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_proc;

    // Local covariance computation
    MatrixXd local_BtB = MatrixXd::Zero(m, m);
    for (int i = start_row; i < end_row; ++i) {
        local_BtB += B.row(i).transpose() * B.row(i);
    }

    // Gather results at the root process
    MatrixXd global_BtB = MatrixXd::Zero(m, m);
    MPI_Reduce(local_BtB.data(), global_BtB.data(), m * m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_BtB /= (n - 1); // Normalize by (n-1)
    }
    return global_BtB;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define dimensions of the matrix
    int n = 1000; // Number of rows
    int m = 1000; // Number of columns

    // Generate random matrix X with values from a standard normal distribution
    MatrixXd X(n, m);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    auto start_time = chrono::high_resolution_clock::now(); // Start timing

    // Only rank 0 generates the matrix
    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                X(i, j) = i + j + dist(gen) * 0.001;
            }
        }
    }

    // Broadcast the matrix X to all processes
    MPI_Bcast(X.data(), n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute mean across columns
    RowVectorXd X_mean(m);
    if (rank == 0) {
        X_mean = X.colwise().mean();
    }
    MPI_Bcast(X_mean.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Subtract mean to center the data
    MatrixXd B = X;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        B.row(i) -= X_mean;
    }

    // Compute variance of the data
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

    // Compute covariance matrix using MPI
    MatrixXd C = mpi_covariance(B, n, m);

    // Perform eigen decomposition (only on rank 0)
    VectorXd eigenvalues;
    MatrixXd eigenvectors;
    if (rank == 0) {
        SelfAdjointEigenSolver<MatrixXd> eigen_solver(C);
        eigenvalues = eigen_solver.eigenvalues(); // Eigenvalues
        eigenvectors = eigen_solver.eigenvectors(); // Eigenvectors
    }

    // Project X onto PC space (only on rank 0)
    MatrixXd T_pca;
    if (rank == 0) {
        T_pca = B * eigenvectors;
    }

    // Perform Singular Value Decomposition (SVD) (only on rank 0)
    JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    VectorXd singular_values = svd.singularValues();
    MatrixXd Vt = svd.matrixV().transpose();

    // Construct the Sigma matrix (only on rank 0)
    MatrixXd Sigma(n, m);
    if (rank == 0) {
        Sigma.setZero();
        for (int i = 0; i < singular_values.size(); ++i) {
            Sigma(i, i) = singular_values(i);
        }
    }

    // Compute T_svd = U * Sigma (only on rank 0)
    MatrixXd T_svd;
    if (rank == 0) {
        T_svd = U * Sigma;
    }

    auto end_time = chrono::high_resolution_clock::now(); // End timing
    chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Save results to files (only on rank 0)
    if (rank == 0) {
        ofstream pca_file("pca_results.txt");
        ofstream svd_file("svd_results.txt");
        ofstream singular_values_file("singular_values.txt");
        ofstream eigenvalues_file("eigenvalues.txt");

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
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

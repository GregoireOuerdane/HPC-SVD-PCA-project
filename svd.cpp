#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;

int main() {
    // Set random seed for reproducibility
    srand(time(0));

    // Define dimensions of the matrix
    int n = 100; // Number of rows
    int m = 100; // Number of columns

    // Generate random matrix X with values from a standard normal distribution
    MatrixXd X(n, m);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    auto start_time = chrono::high_resolution_clock::now(); // Start timing

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            X(i, j) = dist(gen);
        }
    }

    // Compute mean across columns
    RowVectorXd X_mean = X.colwise().mean();

    // Subtract mean to center the data
    MatrixXd B = X;
    for (int i = 0; i < n; ++i) {
        B.row(i) -= X_mean;
    }

    // Compute variance of the data
    double var = (B.array().square()).sum() / (n * m);

    // Normalize the data by dividing by the square root of variance
    B /= sqrt(var);

    // Compute covariance matrix
    MatrixXd C = (B.transpose() * B) / (n - 1);

    // Perform eigen decomposition
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(C);
    VectorXd d = eigen_solver.eigenvalues(); // Eigenvalues
    MatrixXd V = eigen_solver.eigenvectors(); // Eigenvectors

    // Project X onto PC space
    MatrixXd T_pca = B * V;

    // Perform Singular Value Decomposition (SVD)
    JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    VectorXd S = svd.singularValues();
    MatrixXd Vt = svd.matrixV().transpose();

    // Construct the Sigma matrix
    MatrixXd Sigma(n, m);
    Sigma.setZero();
    for (int i = 0; i < S.size(); ++i) {
        Sigma(i, i) = S(i);
    }

    // Compute T_svd = U * Sigma
    MatrixXd T_svd = U * Sigma;

    auto end_time = chrono::high_resolution_clock::now(); // End timing
    chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Save results to files
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
        singular_values_file << S << endl;
        singular_values_file.close();
    }

    if (eigenvalues_file.is_open()) {
        eigenvalues_file << "Eigenvalues of C matrix:\n";
        eigenvalues_file << d << endl;
        eigenvalues_file.close();
    }

    // Print execution time
    cout << "Execution time: " << elapsed_seconds.count() << " seconds" << endl;

    return 0;
}

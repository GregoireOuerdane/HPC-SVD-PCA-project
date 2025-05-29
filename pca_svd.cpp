#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <algorithm>
#include <cstddef>  // For size_t, ptrdiff_t
#include <cstdlib>  // For nullptr_t

using namespace std;
using namespace std::chrono;

class ParallelPCA {
private:
    vector<vector<double>> data;
    vector<vector<double>> centered_data;
    vector<vector<double>> covariance_matrix;
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    vector<double> singular_values;
    vector<vector<double>> U; // left singular vectors
    vector<vector<double>> V; // right singular vectors (PCA components)
    int n_samples;
    int n_features;

public:
    ParallelPCA(int samples, int features) : n_samples(samples), n_features(features) {
        data.resize(n_samples, vector<double>(n_features));
        centered_data.resize(n_samples, vector<double>(n_features));
        covariance_matrix.resize(n_features, vector<double>(n_features));
        eigenvectors.resize(n_features, vector<double>(n_features));
        U.resize(n_samples, vector<double>(n_features));
        V.resize(n_features, vector<double>(n_features));
    }

    // generate random data matrix
    void generate_random_data() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 1.0);

        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                data[i][j] = dist(gen);
            }
        }
    }

    // center the data by subtracting the mean of each feature
    void center_data() {
        vector<double> means(n_features, 0.0);

        // compute means (parallel reduction)
        #pragma omp parallel for
        for (int j = 0; j < n_features; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                sum += data[i][j];
            }
            means[j] = sum / n_samples;
        }

        // subtract means (parallel)
        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                centered_data[i][j] = data[i][j] - means[j];
            }
        }

        // check the column means are ~0 after centering
        //for (int j = 0; j < n_features; j++) {
        //    double col_sum = 0;
        //    for (int i = 0; i < n_samples; i++) {
        //        col_sum += centered_data[i][j];
        //    }
        //    cout << "Column " << j << " mean: " << col_sum/n_samples << endl;
        //}
    }

    // compute covariance matrix in parallel
    void compute_covariance_matrix() {
        // parallel matrix multiplication: X^T * X
        #pragma omp parallel for
        for (int i = 0; i < n_features; ++i) {
            for (int j = 0; j < n_features; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n_samples; ++k) {
                    sum += centered_data[k][i] * centered_data[k][j];
                }
                covariance_matrix[i][j] = sum / (n_samples - 1);
            }
        }

        // diagonal should have positive values
        //for (int i = 0; i < n_features; i++) {
        //    cout << "Cov[" << i << "][" << i << "] = " 
        //         << covariance_matrix[i][i] << endl;
        //}
    }

    // power iteration method for finding eigenvectors
    void power_iteration(int max_iter = 1000, double tol = 1e-8) {
        eigenvalues.resize(n_features);
        eigenvectors.resize(n_features, vector<double>(n_features, 0));
    
        vector<vector<double>> A = covariance_matrix;
    
        for (int i = 0; i < n_features; ++i) {
            vector<double> v(n_features, 1.0); // initial guess
            double lambda = 0.0;
        
            for (int iter = 0; iter < max_iter; ++iter) {
                // matvec
                vector<double> Av(n_features, 0.0);
                #pragma omp parallel for
                for (int row = 0; row < n_features; ++row) {
                    double sum = 0.0;
                    for (int col = 0; col < n_features; ++col) {
                        sum += A[row][col] * v[col];
                    }
                    Av[row] = sum;
                }
            
                // compute new eigenvalue
                double lambda_new = 0.0;
                #pragma omp parallel for reduction(+:lambda_new)
                for (int j = 0; j < n_features; ++j) {
                    lambda_new += Av[j] * v[j];
                }
            
                // normalize
                double norm = 0.0;
                #pragma omp parallel for reduction(+:norm)
                for (int j = 0; j < n_features; ++j) {
                    norm += Av[j] * Av[j];
                }
                norm = sqrt(norm);
                
                #pragma omp parallel for
                for (int j = 0; j < n_features; ++j) {
                    v[j] = Av[j] / norm;
                }
            
                // check convergence
                if (fabs(lambda_new - lambda) < tol) {
                    break;
                }
                lambda = lambda_new;
            }
        
            eigenvalues[i] = lambda;
            for (int j = 0; j < n_features; ++j) {
                eigenvectors[j][i] = v[j];
            }
        
            // deflate the matrix
            #pragma omp parallel for
            for (int row = 0; row < n_features; ++row) {
                for (int col = 0; col < n_features; ++col) {
                    A[row][col] -= lambda * v[row] * v[col];
                }
            }
        }
    
        // check eigenvalues are positive
        //for (int i = 0; i < n_features; ++i) {
        //    eigenvalues[i] = fabs(eigenvalues[i]);
        //}

        //cout << "Top 10 eigenvalues:\n";
        //for (int i = 0; i < 10; ++i) {
        //    cout << eigenvalues[i] << " ";
        //}
        //cout << "\n";
    }

    // SVD implementation with power iteration
    void compute_svd(int max_iter = 1000, double tol = 1e-8) {
        singular_values.resize(min(n_samples, n_features));
        
        // compute right singular vectors (V) which are PCA components
        power_iteration(max_iter, tol);
        V = eigenvectors;
        
        // compute singular values from eigenvalues
        #pragma omp parallel for
        for (int i = 0; i < singular_values.size(); ++i) {
            singular_values[i] = sqrt(eigenvalues[i]);
        }
        
        // compute left singular vectors (U)
        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                U[i][j] = 0.0;
                for (int k = 0; k < n_features; ++k) {
                    U[i][j] += centered_data[i][k] * V[k][j];
                }
                if (j < singular_values.size() && singular_values[j] > 1e-10) {
                    U[i][j] /= singular_values[j];
                }
            }
        }
    }

    // project data onto principal components
    vector<vector<double>> transform(int n_components) {
        n_components = min(n_components, n_features);
        vector<vector<double>> transformed(n_samples, vector<double>(n_components));
        
        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_components; ++j) {
                transformed[i][j] = 0.0;
                for (int k = 0; k < n_features; ++k) {
                    transformed[i][j] += centered_data[i][k] * V[k][j];
                }
            }
        }
        
        return transformed;
    }

    // save results to files
    void save_results(const string& prefix) {
        // save eigenvalues
        ofstream eval_file(prefix + "_eigenvalues.txt");
        for (double val : eigenvalues) {
            eval_file << val << "\n";
        }
        eval_file.close();

        // save singular values
        ofstream sval_file(prefix + "_singular_values.txt");
        for (double val : singular_values) {
            sval_file << val << "\n";
        }
        sval_file.close();

        // save transformed data (first 2 components)
        auto transformed = transform(2);
        ofstream pca_file(prefix + "_pca_results.txt");
        for (const auto& row : transformed) {
            pca_file << row[0] << " " << row[1] << "\n";
        }
        pca_file.close();
    }

    void save_matrices(const string& prefix, const std::chrono::milliseconds& duration) {
        // Save data matrix
        ofstream data_file(prefix + "_data.txt");
        for (const auto& row : data) {
            for (double val : row) data_file << val << " ";
            data_file << "\n";
        }
        data_file.close();

        // Save covariance matrix
        ofstream cov_file(prefix + "_cov.txt");
        for (const auto& row : covariance_matrix) {
            for (double val : row) cov_file << val << " ";
            cov_file << "\n";
        }
        cov_file.close();

        // Save runtime (in milliseconds)
        ofstream time_file(prefix + "_time.txt");
        time_file << duration.count();
        time_file.close();
    }
};

int main() {
    const int n_samples = 1000;
    const int n_features = 100;
    
    // set number of threads for OpenMP
    omp_set_num_threads(4);
    
    ParallelPCA pca(n_samples, n_features);
    
    auto start = high_resolution_clock::now();
    
    // generate random data
    pca.generate_random_data();
    
    pca.center_data();
    
    pca.compute_covariance_matrix();
    
    // compute PCA (eigen decomposition)
    pca.power_iteration();
    
    pca.compute_svd();
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    cout << "Execution time: " << duration.count() << " ms" << endl;
    
    // save results for visualization
    pca.save_results("parallel_pca");
    pca.save_matrices("cpp_pca", duration);
    
    return 0;
}
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;

VectorXd unfoldDim2(const MatrixXd& X);
VectorXd generate_permutation_vector(const VectorXd& nFolded, const VectorXd& mFolded);
MatrixXd permutate(const VectorXd& permutation_vector, const MatrixXd& X);
MatrixXd foldDim0(const VectorXd& vect_representation, int dim0, int dim1, int dim2);
MatrixXd foldDim1(const VectorXd& vect_representation, int dim0, int dim1, int dim2);
MatrixXd foldDim2(const VectorXd& vect_representation, int dim0, int dim1, int dim2);
MatrixXd maxpool_backprop(const MatrixXd& X, const MatrixXd& Y, const MatrixXd& dY);
MatrixXd updateParameters(const MatrixXd& X, const MatrixXd& dX);
MatrixXd kronecker_product(const MatrixXd& A, const MatrixXd& B);
int argmax(const VectorXd& vect);
MatrixXd maxpool(const MatrixXd& input);
MatrixXd convolution(const MatrixXd& X, const MatrixXd& K);
MatrixXd elementwise_product(MatrixXd X1, MatrixXd X2);
MatrixXd relu(MatrixXd mat);
MatrixXd relu_prime(MatrixXd mat);
MatrixXd dot(const MatrixXd& X1, const MatrixXd& X2);
MatrixXd readCSV(const std::string& filePath);
void saveCSV(const MatrixXd& matrix, const std::string& filename);
MatrixXd vectorToMatrix(const VectorXd& vect, int dim1, int dim2);

#endif // FUNCTIONS_H
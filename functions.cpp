#include "functions.h"
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

const float eta = 0.001; // akin to learning rate but less sensitive
const float delta = 0.0000001; //small constant used to prevent division by 0 

using namespace Eigen;

VectorXd unfoldDim2(const MatrixXd& X){
    /**
    * @brief Unfolds matrix along the second dimension and saves to vector. 
    *  Essentially taking each row and concatenating. 
    *
    * @param X: input matrix 
    * @return unfolded representation of input matrix in vector form
    */
    VectorXd unfolded = VectorXd::Zero(X.rows() * X.cols()); 
    for(int i = 0; i < X.rows(); i++){
        unfolded.segment(X.cols() * i, X.cols()) = X.row(i);
    }
    return unfolded; 
}

void saveCSV(const MatrixXd& matrix, const std::string& filename){
    /**
    * @brief saves a matrix to a given filename in csv format
    *
    * @param matrix: input matrix 
    * @param filename: string that ends in .csv
    * @return none 
    */
    std::ofstream file(filename);
    if (file.is_open()) {
        file << matrix.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"));
        file.close();
        std::cout << "Matrix saved to " << filename << std::endl;
    } else {std::cerr << "Error opening file: " << filename << std::endl;}
}

MatrixXd vectorToMatrix(const VectorXd& vect, int dim0, int dim1){
    /**
    * @brief converts a vector to a matrix given dimensions of matrix 
    * elements are added to matrix column by column
    *
    * @param vect: input vector 
    * @param dim0: length of first dimension of new matrix 
    * @param dim1: length of second dimension of new matrix 
    * @return matrix representation of vector
    */
    MatrixXd mat = MatrixXd::Zero(dim0, dim1);
    for(int j = 0; j < dim1; j++){
    for(int i = 0; i < dim0; i++){
        mat(i,j) = vect(i%dim0 + j*dim0); 
    }}
    return mat;
}

MatrixXd permutate(const VectorXd& permutation_vector, const MatrixXd& X){
    /**
    * @brief rows of input matrix are rearranged given a permutation vector
    *
    * @param permutation_vector: permutation_vector[i] = j means row i of X is moved to row j  
    * @param X: input matrix
    * @return permutated matrix 
    */
    MatrixXd permutated = MatrixXd::Zero(X.rows(),X.cols());
    for(int i = 0; i < X.rows(); i++){
        permutated.row(permutation_vector(i)) = X.row(i);
    }
    return permutated;
}

VectorXd generate_permutation_vector(const VectorXd& nFolded, const VectorXd& mFolded){
    /**
    * @brief given 2 vectorized foldings along different dimensions of a weight matrix, 
    *        permutation vector is generated that maps one to the other
    *
    * @param nFolded: matrix folded along nth dimension flattened
    * @param mFolded: matrix folded along mth dimension flattened
    * @return permutation vector 
    */
    VectorXd permutation_vector(nFolded.size());
    for(int i = 0; i < nFolded.size(); i++){
        for(int j = 0; j < mFolded.size(); j++){
            if(nFolded(i) == mFolded(j)){
                permutation_vector(i) = j;
    }}}
    return permutation_vector;
}

MatrixXd maxpool_backprop(const MatrixXd& X, const MatrixXd& Y, const MatrixXd& dY){
    /**
    * @brief propagates gradient backward through maxpooling layer
    *
    * @param X: input of maxpool during forward propagation
    * @param Y: output of maxpool during forward propagation
    * @param dY: gradient at output of maxpool during backward propagation
    * @return gradient at input of maxpool during backward propagation
    */
    MatrixXd backpropagated = MatrixXd::Zero(X.rows(),X.cols());
    for(int i = 0; i < X.rows(); i++){
        for(int j = 0; j < X.cols(); j++){
            if(pow(X(i,j) - Y(i/2,j/2),2) < 0.001){ backpropagated(i,j) = dY(i/2,j/2);} 
    }}
    return backpropagated;
}

MatrixXd foldDim2(const VectorXd& vect_representation, int dim0, int dim1, int dim2){
    /**
    * @brief folds a tensor along dimension 2
    *
    * @param vect_representation: flattened representation of a tensor 
    * @param dim0: length of dimension 0 of tensor 
    * @param dim1: length of dimension 1 of tensor 
    * @param dim2: length of dimension 2 of tensor
    * @return matrix representation of tensor folded along dimension 2
    */
    MatrixXd folded_representation(dim2, dim0*dim1);
    for(int i = 0; i < dim2; i++){
    for(int j = 0; j < dim0*dim1; j++){ 
        folded_representation(i,j) = vect_representation(j % (dim0*dim1) + (dim0*dim1) * i); 
    }}
    return folded_representation;
}

MatrixXd foldDim1(const VectorXd& vect_representation, int dim0, int dim1, int dim2){
    /**
    * @brief folds a tensor along dimension 1
    *
    * @param vect_representation: flattened representation of a tensor 
    * @param dim0: length of dimension 0 of tensor 
    * @param dim1: length of dimension 1 of tensor 
    * @param dim2: length of dimension 2 of tensor
    * @return matrix representation of tensor folded along dimension 1
    */
    MatrixXd folded_representation(dim1,dim0*dim2);
    for(int i = 0; i < dim1; i++){
    for(int j = 0; j < dim0*dim2; j++){ //needs correction
        folded_representation(i,j) = vect_representation(j%dim1 + dim0*dim1*(j/dim1) + dim1*i);
    }}
    return folded_representation;
}

MatrixXd foldDim0(const VectorXd& vect_representation, int dim0, int dim1, int dim2) {
    /**
    * @brief folds a tensor along dimension 0
    *
    * @param vect_representation: flattened representation of a tensor 
    * @param dim0: length of dimension 0 of tensor 
    * @param dim1: length of dimension 1 of tensor 
    * @param dim2: length of dimension 2 of tensor
    * @return matrix representation of tensor folded along dimension 0
    */
    MatrixXd folded_representation(dim0,dim1*dim2);
    for(int i = 0; i < dim0; i++){
    for(int j = 0; j < dim0*dim1; j++){
        folded_representation(i,j) = vect_representation(dim0*j+i);
    }}
    return folded_representation;
}

MatrixXd updateParameters(const MatrixXd& X, const MatrixXd& dX){
    /**
    * @brief updates the parameters represented in matrix form
    * adjustments are multiplied by a factor eta times the original parameters
    *
    * @param X: matrix of parameters 
    * @param dX: partial derivative of parameters with respect to loss
    * @return updated parameter matrix  
    */
    float multiplier = eta*X.norm()/(dX.norm() + delta);
    return X - multiplier*dX;
}

MatrixXd readCSV(const std::string& filePath) {
    /**
    * @brief reads the contents of a csv file and writes to a MatrixXd
    *
    * @param filePath: filepath of matrix contents
    * @return matrix 
    */

    // Open the CSV file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }
    std::cout << "Reading: " << filePath;
    // Read data from the CSV file
    std::vector<std::vector<float>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream lineStream(line);
        float value;
        while (lineStream >> value) {
            row.push_back(value);
            if (lineStream.peek() == ',') {
                lineStream.ignore();
            }
        }
        data.push_back(row);
    }

    // Close the file
    file.close();

    // Convert data to Eigen::MatrixXd
    int numRows = data.size();
    int numCols = data[0].size();
    MatrixXd matrix(numRows, numCols);
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    std::cout << "\tFinished "<< std::endl;
    return matrix;
}

MatrixXd kronecker_product(const MatrixXd& A, const MatrixXd& B) {
    /**
    * @brief performs kronecker product between two matrices 
    *
    * @param A: matrix
    * @param B: matrix
    * @return matrix 
    */

    MatrixXd result(A.rows() * B.rows(), A.cols() * B.cols());

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return result;
}

int argmax(const VectorXd& vect){
    /**
    * @brief determines the index of the maximum element of a VectorXd, 
    * first if there are duplicate maximums
    *
    * @param vect: vector
    * @return index of maximum element of vector  
    */
    float max = vect.maxCoeff();
    for(int i = 0; i < vect.size(); i++){
        if(pow(vect(i) - max,2) < 0.0001){ return i; }
    }
}

MatrixXd maxpool(const MatrixXd& input) {
    /**
    * @brief forward propagation of maxpool layer
    *
    * @param input: input matrix which should represent image or feature map
    * @return matrix of compressed input  
    */
    MatrixXd pooledMatrix(input.rows()/2, input.cols()/2);
    for (int i = 0; i < input.rows()/2; ++i) {
        for (int j = 0; j < input.cols()/2; ++j) {
            MatrixXd window = input.block(i * 2, j * 2, 2, 2);
            double maxValue = window.maxCoeff();
            pooledMatrix(i, j) = maxValue;
    }}
    return pooledMatrix;
}

MatrixXd convolution(const MatrixXd& X, const MatrixXd& K){
    /**
    * @brief forward propagation of convolution layer
    *
    * @param X: input matrix
    * @param K: kernel matrix
    * @return feature map matrix 
    */
    int output_dim = X.rows() - K.rows() + 1;
    MatrixXd Y = MatrixXd::Zero(output_dim, output_dim);
    for(int i = 0; i < output_dim; i++){
        for(int j = 0; j < output_dim; j++){
            for(int u = 0; u < K.rows(); u++){
                for(int v = 0; v < K.rows(); v++){
                    Y(i,j) += X(i+u,j+v) * K(u,v);
    }}}}
    return Y;
}

MatrixXd relu(MatrixXd mat) {
    /**
    * @brief computes the relu operation on an input matrix
    *
    * @param mat: input matrix
    * @return relu matrix 
    */
    MatrixXd relu_matrix = MatrixXd::Zero(mat.rows(),mat.cols());
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            relu_matrix(i, j) = std::max(0.0, mat(i, j));
        }
    }
    return relu_matrix;
}

MatrixXd elementwise_product(MatrixXd X1, MatrixXd X2){ 
    /**
    * @brief computes the elementwise product between two input matrices
    * matrices must have same dimenstions 
    *
    * @param X1: matrix
    * @param X2: matrix
    * @return elementwise product matrix
    */
    MatrixXd prod = MatrixXd::Zero(X1.rows(), X1.cols()); 
    for (int i = 0; i < X1.rows(); ++i) {
        for (int j = 0; j < X1.cols(); ++j) {
            prod(i,j) = X1(i,j) * X2(i,j);
    }}
    return prod;
}

MatrixXd relu_prime(MatrixXd mat){
    /**
    * @brief computes the derivative of relu given the input matrix of the computation 
    *
    * @param mat: input matrix
    * @return derivative of relu matrix 
    */
    MatrixXd relu_prime = MatrixXd::Zero(mat.rows(),mat.cols());
    for(int i = 0; i < mat.rows(); i++){
        for(int j = 0; j < mat.cols(); j++){
            if(mat(i,j) > 0){ relu_prime(i,j) = 1; } 
    }}
    return relu_prime;
}

MatrixXd dot(const MatrixXd& X1, const MatrixXd& X2){
    /**
    * @brief computes the dot product (inner product) of two matrices 
    * this should be substituted with built in eigen library dot product function 
    * but I was experiencing bugs when I used larger input matrices
    *
    * @param X1: matrix
    * @param X2: matrix
    * @return matrix  
    */
    if(X1.cols() != X2.rows()){ std::cout << "X1.cols() != X2.rows()" << std::endl; }
    MatrixXd dot_product = MatrixXd::Zero(X1.rows(),X2.cols());
    for(int i = 0; i < X1.rows(); i++){
        for(int j = 0; j < X2.cols(); j++){
            for(int k = 0; k < X1.cols(); k++){ dot_product(i,j) += X1(i,k)*X2(k,j); }
    }}
    return dot_product;
}
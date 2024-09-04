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

#include "functions.h"
using namespace Eigen;

//HYPER PARAMETERS 
const int filters = 32; // MUST BE 32 FOR CORRECT TENSOR SHAPES, NAMED 'filters' FOR CODE READABILITY
const int update_parameters = 1;    //FOR TESTING PURPOSES THESE CAN BE SET TO 0 
const int save_parameters = 1;      //


int main(int argc, char* argv[]){
    std::string input_images_filename = "datasets/mini.csv"; // CSV FILE THAT CONTAINS TRAINING IMAGES

    //PROPAGATION STORAGE 
    Matrix<double,3,3> KERNEL[filters];                 //FORWARD PROPAGATION VALUES
    Matrix<double,26,26> CONV_OUTPUT[filters];          //
    Matrix<double,26,26> RELU_OUTPUT[filters];          //
    Matrix<double,13,13> POOLING_OUTPUT[filters];       //

    Matrix<double,3,3> del_KERNEL[filters];             //BACKWARD PROPAGATION VALUES
    Matrix<double,26,26> del_CONV_OUTPUT[filters];      //
    Matrix<double,26,26> del_RELU_OUTPUT[filters];      //
    Matrix<double,13,13> del_POOLING_OUTPUT[filters];   //

    //CALCULATING PERMUTATION VECTORS TO BE USED IN BACKPROPAGATION
    //SEE PAPER FOR DETAILS
    VectorXd practice = VectorXd::Zero(2000); 
    for(int i = 0; i < 2000; i++) { practice(i) = i; }
    MatrixXd P_0 = foldDim0(practice,10,10,20);
    MatrixXd P_1 = foldDim1(practice,10,10,20);
    MatrixXd P_2 = foldDim2(practice,10,10,20);
    VectorXd PERMUTATION0 = generate_permutation_vector(P_0.reshaped(), P_2.reshaped());
    VectorXd PERMUTATION1 = generate_permutation_vector(P_1.reshaped(), P_2.reshaped());
    VectorXd PERMUTATION2 = generate_permutation_vector(P_2.reshaped(), P_2.reshaped());
    
    //READING LABELS (int) AND IMAGES (28 x 28 x 1)
    MatrixXd csvdata = readCSV(input_images_filename);
    VectorXd labels = csvdata.col(0); //VECTOR OF DIGITS FROM 0 TO 9 THAT REPRESENT LABEL OF IMAGES
    MatrixXd images(csvdata.rows(), csvdata.cols() - 1);
    images << csvdata.rightCols(csvdata.cols() - 1); // MATRIX WHERE EACH ROW IS A FLATTENED REPRESENTATION OF IMAGE MATRIX
    
    //READING PARAMETERS 
    for(int i = 0; i < filters; i++){           // KERNELS
        KERNEL[i] = readCSV("kernels/K"+std::to_string(i)+".csv");}
    MatrixXd U0 = readCSV("tensors/U0.csv");    // MATRICES FOR DECOMPOSITION OPERATION
    MatrixXd U1 = readCSV("tensors/U1.csv");    // 
    MatrixXd U2 = readCSV("tensors/U2.csv");    //
    MatrixXd W = readCSV("tensors/W.csv");              // WEIGHTS FOR LINEAR LAYER

    //TRAINING LOOP SPECIFICATIONS 
    const int trials = 1000;                // HOW MANY SGD TRIALS
    const int desired_sample_size = 60000;  // DESIRED NUMBER OF IMAGES TO TRAIN ON
    const int save_frequency = 100;         // PARAMETERS BE SAVED TO CSV FILES EVERY ___ TRIALS 
    const int measurement_duration = 100;   // PREVIOUS ___ TRIALS USED TO MEASURE ERROR AND ACCURACY
    const int sample_size = std::min(desired_sample_size, images.rows()); 

    //MEASUREMENT VECTORS 
    VectorXd acc = VectorXd::Zero(measurement_duration); // STORES ACCURACY OF PREVIOUS TRIALS
    VectorXd err = VectorXd::Zero(measurement_duration); // STORES ERROR OF PREVIOUS TRIALS

    //TRAINING LOOP 
    for(int trial = 0; trial < trials; trial++){
        //FETCHING INPUT IMAGE AND LABEL
        int idx = trial%sample_size;

        //INCORRECT PREDICTIONS ARE LABELED 5, CORRECT LABELED 10. 
        //THIS CAN BE CHOSEN ARBITRARY BUT MUST BE ABLE TO DIFFERENTIATE. 
        VectorXd Y_label = VectorXd::Constant(10, 5); Y_label(static_cast<int>(labels(idx))) = 10;

        //INPUT IMAGE IS NORMALIZED TO HAVE ZERO MEAN FOR MORE STABLE TRAINING
        MatrixXd input_image_unnormalized = images.row(idx).reshaped(28,28).transpose();
        MatrixXd input = input_image_unnormalized/input_image_unnormalized.norm();

        //FORWARD PROPAGATION 
        for(int i = 0; i < filters; i++){
            CONV_OUTPUT[i] = convolution(input, KERNEL[i]); // PERFORM CONVOLUTION
            RELU_OUTPUT[i] = relu(CONV_OUTPUT[i]);          // PERFORM RELU 
            POOLING_OUTPUT[i] = maxpool(RELU_OUTPUT[i]);    // PERFORM MAXPOOLING 
        }


        
        //TENSOR DECOMPOSITION STAGE START /////////////////////////////////////////////////////////
        VectorXd X = VectorXd::Zero(13*13*filters); // REPRESENTS INPUT TENSOR TO DECOMPOSITION
        for(int i = 0; i < filters; i++){ X.segment(13*13*i,13*13) = POOLING_OUTPUT[i].reshaped();}

        //FOLD ALONG SECOND DIMENSION
        MatrixXd X2 = foldDim2(X,13,13,32); 

        //COMPUTE CORE TENSOR (G) AS DESCRIBED IN SOURCE PAPER 
        MatrixXd G2 = dot(dot(U2,X2),kronecker_product(U1,U0).transpose());
        
        //RESHAPE TO BE VECTOR 
        VectorXd G2_vect = G2.reshaped();    
        //TENSOR DECOMPOSITION STAGE END ///////////////////////////////////////////////////////////



        //APPLY LINEAR LAYER TO CORE TENSOR OF DECOMPOSITION STAGE
        VectorXd Y_pred = W*G2_vect;

        //MEASURING ACCURACY OF PREDICTION
        int correct = 0; 
        if(argmax(Y_pred) == argmax(Y_label)) { correct = 1; } 
        acc(trial%measurement_duration) = correct; 
    
        //COMPUTING DERIVATIVE OF LOSS WITH RESPECT TO PREDICTION
        VectorXd del_Y_pred = Y_pred - Y_label;

        //MEASURING ERROR OF PREDICTION 
        err(trial%measurement_duration) = del_Y_pred.norm();

        //ENSURING MEASUREMENTS ARE REASONABLE WHEN MEASUREMENT VECTORS NOT FULL YET (CAN BE IGNORED)
        if(trial < measurement_duration) { 
            for(int i = trial; i < measurement_duration; i++){ 
                err(i) = err(trial); 
                //acc(i) = acc(trial);
            } 
        }

        //DISPLAYING RESULTS 
        std::cout << "TRIAL: " << trial << "\tERR: " << del_Y_pred.norm()  
        << "\tACCURACY AVG: " << acc.mean() << "\tERROR AVG: " << err.mean() << std::endl;



        //BEGINNING OF BACK PROPAGATION 

        //DERIVATIVE OF LOSS WITH RESPECT TO FULLY CONNECTED LAYER WEIGHTS
        MatrixXd dW = MatrixXd::Zero(W.rows(),W.cols());
        for(int i = 0; i < W.rows(); i++){ 
        for(int j = 0; j < W.cols(); j++){ 
            dW(i,j) = del_Y_pred(i)*G2_vect(j); 
        }}

        //DERIVATIVE OF LOSS WITH RESPECT TO CORE TENSOR OF DECOMPOSITION
        VectorXd del_G = VectorXd::Zero(W.cols());
        for(int i = 0; i < W.cols(); i++){ 
        for(int j = 0; j < W.rows(); j++){ 
            del_G(i) = del_G(i) + del_Y_pred(j)*W(j,i);
        }}




        //TENSOR DECOMPOSITION LAYER BACK PROPAGATION ///////////////////////////////////////////////
        MatrixXd del_G2 = vectorToMatrix(del_G,20,100);

        //PARTIAL DERIVATIVE OF CORE TENSOR WITH RESPECT TO INPUT TENSOR
        MatrixXd DG2DX2 = kronecker_product(kronecker_product(U1,U0),U2);

        //PARTIAL DERIVATIVE OF LOSS WITH RESPECT TO INPUT TENSOR
        MatrixXd DLDX2 = vectorToMatrix(DG2DX2.transpose() * del_G2.reshaped(), 32,169);
        VectorXd DLDX = unfoldDim2(DLDX2);

        //PARTIAL DERIVATIVES OF MATRICES INVOLVED IN DECOMPOSITION WITH RESPECT TO CORE TENSOR AS DESCRIBED IN PAPER
        MatrixXd DGDU2 = permutate(PERMUTATION2, kronecker_product(dot(kronecker_product(U1,U0),foldDim2(X,13,13,32).transpose()),MatrixXd::Identity(20, 20)));
        MatrixXd DGDU1 = permutate(PERMUTATION1, kronecker_product(dot(kronecker_product(U2,U0),foldDim1(X,13,13,32).transpose()),MatrixXd::Identity(10, 10)));
        MatrixXd DGDU0 = permutate(PERMUTATION0, kronecker_product(dot(kronecker_product(U2,U1),foldDim0(X,13,13,32).transpose()),MatrixXd::Identity(10, 10)));        

        //PARTIAL DERIVATIVES OF MATRICES INVOLVED IN DECOMPOSITION WITH RESPECT TO LOSS
        MatrixXd del_U2 = dot(DGDU2.transpose(), del_G).reshaped(20,32);
        MatrixXd del_U1 = dot(DGDU1.transpose(), del_G).reshaped(10,13);
        MatrixXd del_U0 = dot(DGDU0.transpose(), del_G).reshaped(10,13);

        //TENSOR DECOMPOSITION LAYER BACK PROPOGATION ///////////////////////////////////////////////



        //BACKPROPAGATION OF REMAINING LAYERS 
        for(int i = 0; i < filters; i++){
            del_POOLING_OUTPUT[i] = DLDX.segment(i*169,169).reshaped(13,13); 
            del_RELU_OUTPUT[i] = maxpool_backprop(CONV_OUTPUT[i],POOLING_OUTPUT[i],del_POOLING_OUTPUT[i]); 
            del_CONV_OUTPUT[i] = elementwise_product(relu_prime(CONV_OUTPUT[i]),del_RELU_OUTPUT[i]);
            del_KERNEL[i] = convolution(input, del_CONV_OUTPUT[i]); 
        }
        
        //UPDATING PARAMETERS USING STOCHASTIC GRADIENT DESCENT
        W = updateParameters(W,dW); 
        U0 = updateParameters(U0,del_U0);
        U1 = updateParameters(U1,del_U1);
        U2 = updateParameters(U2,del_U2);
        for(int i = 0; i < filters; i++){ KERNEL[i] = updateParameters(KERNEL[i],del_KERNEL[i]); }

        //SAVING WEIGHTS TO CSV FILES FOR STORAGE
        if(trial%save_frequency == save_frequency-1){
            saveCSV(W,"W.csv");
            saveCSV(U0,"tensors/U0.csv");
            saveCSV(U1,"tensors/U1.csv");
            saveCSV(U2,"tensors/U2.csv");
            for(int i = 0; i < filters; i++){ 
                saveCSV(KERNEL[i],"kernels/K"+std::to_string(i)+".csv");
            }
        }
    }

    return 0; 
}
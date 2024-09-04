# TCL_CPP
Tensor Contraction Layer backpropagation implementation in C++.

**Dependencies:** 
- g++ 

**Steps to Train Parameters:**
1. extract contents from zip file `eigen-3.4.0.zip` and save to folder titled `eigen-3.4.0`
2. run ` g++ main.cpp -O3 functions.cpp "-Ieigen-3.4.0"` in command prompt
3. run `./a.exe`
4. observe changes in average accuracy and average error
    1. Note 1: the save_frequency variable is set to 100 so the first 100 results will be deflated because values initialized to 0
    2. Note 2: the weights have already been trained so accuracy should be relatively stable at approx 85-90%

**Hyperparameters:**
- `update_parameters` controls if training takes place, set to 0 for pure testing without training 
- `save_parameters` controls if parameters are saved to csv files periodically 
- `trials` controls number of iterations of training loop 
- `desired_sample_size` allows user to enter maximum sample size 
- `sample_size` is the minimum of the `desired_sample_size` and number of images found in training set
- `save_frequency` controls the number of training iterations before parameters are saved to csv files
- `measurement_duration` controls the number of previous trials involved in avg accuracy and avg error calculations

**Additional Notes:** 
- Weights can be set to random initial values by running the script `parameter_randomizer.py`
- The file `datasets/mini.csv` includes 2000 MNIST samples which is a relatively small amount but loading the original 60K can take time. 
- Original MNIST dataset can be found at https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download 



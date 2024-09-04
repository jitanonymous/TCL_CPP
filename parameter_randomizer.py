import numpy as np

def GENMAT(FILENAME, ROWS, COLS,LOWER_BOUND=-0.01, UPPER_BOUND=0.01):
    random_matrix = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(ROWS, COLS))
    np.savetxt(FILENAME, random_matrix, delimiter=',')

def GEN_K(filename, rows,columns):
    random_matrix = np.random.uniform(-0.01, 0.01, size=(rows, columns))
    random_matrix[1][1] = 1
    np.savetxt(filename, random_matrix, delimiter=',')

if __name__ == "__main__":
    #RUN SCRIPT TO INITIALIZE RANDOM MODEL PARAMETERS FOR TRAINING
    filters = 32
    for i in range(filters):
        GEN_K("kernels/iK"+str(i)+".csv",3,3)
    GENMAT("tensors/iU0.csv", 10,13,0,0.01)
    GENMAT("tensors/iU1.csv", 10,13,0,0.01)
    GENMAT("tensors/iU2.csv", 20,32,0,0.01)
    GENMAT("tensors/iW.csv", 10,2000,0,0.01)








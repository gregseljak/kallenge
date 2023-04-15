### This is largely inspired by HW2 but adapted to our needs

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy as cp
import time
import pandas as pd
import copy
from scipy.sparse import load_npz

# Note that we only use a distance computation function from sklearn (for sparse matrices), no learning agorithm from sklearn is used.
from sklearn.metrics import pairwise_distances


class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
        
    def kernel(self,X,Y, indexes_of_change_size_subgraphs):
        #### equilibrated min_max kernel
        ## Input vectors X and Y of shape Nxd and Mxd

        result_kernel = np.zeros((X.shape[0], Y.shape[0]))
        indexes_of_change_size_subgraphs = list(indexes_of_change_size_subgraphs)
        indexes_of_change_size_subgraphs.append(X.shape[1])
        for i in range(len(indexes_of_change_size_subgraphs) - 1) :
            x = X[:,indexes_of_change_size_subgraphs[i]: indexes_of_change_size_subgraphs[i+1]]
            y = Y[:,indexes_of_change_size_subgraphs[i]: indexes_of_change_size_subgraphs[i+1]]
            average = (x.sum(axis=-1) + y.sum(axis=-1).T)/2.
            difference =  pairwise_distances(x,y, metric =  "cityblock")/2
        
            max = average + difference
            min = average - difference
            min_max_kernel = min/(max + 0.00000000001)
            result_kernel += min_max_kernel
        return result_kernel/(len(indexes_of_change_size_subgraphs))


class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.y = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y, indexes_of_change_size_subgraphs):
       #### You might define here any v ariable needed for the rest of the code
        N = len(y)
        y = copy.deepcopy(y)
        y[y==0] = -1

        self.y = y
        self.X = X
        self.support = None
        print("start kernel calculation at", time.time())
        K = self.kernel(X,X, indexes_of_change_size_subgraphs)
        print("finish at", time.time())

        ## IMPORTANCE PONDERATION !!
        ponderations = copy.deepcopy(y)
        ponderations[ponderations == 1] = 5455./555.
        # ponderations[ponderations == 1] = 1.0
        ponderations[ponderations == -1] = 1.0

        ### SECOND version of minimization : FAST ? 
        print("start minimization at ", time.time())
        y = y.reshape((-1,1))
        ponderated_y = ponderations.reshape((-1,1))*y
        ponderations = ponderations.reshape((-1,1))
        n = K.shape[0]
        alpha = cp.Variable((n, 1))
        K_ =  cp.Parameter(K.shape, PSD=True)
        constraints = [cp.multiply(y,alpha) >= 0, cp.multiply(y,alpha) <= self.C]

        objective = cp.Minimize( 0.5*cp.quad_form(cp.multiply(alpha,ponderations),K_)   - cp.sum(cp.multiply(ponderated_y,alpha) )   ) # 0.5*beta.reshape((1,-1))@K@beta - np.sum(alpha)
        problem = cp.Problem(objective, constraints)
        # assert problem.is_dpp() # is not important for us, we optimize only once
        K_.value = K
        problem.solve()
        
        alpha = problem.variables()[0].value * ponderated_y
        self.alpha = alpha.flatten()
        y = y.flatten()
        print("end minimization at ", time.time())
        ponderations = ponderations.flatten()
        ## Assign the required attributes
        print("alpha")
        for a in alpha :
            print(a)
        indices = np.nonzero( (self.alpha > self.epsilon*ponderations)*( self.alpha < ponderations*(self.C - self.epsilon)) )[0]
        print(np.nonzero( (self.alpha > self.epsilon*ponderations)*( self.alpha < ponderations*(self.C - self.epsilon))))
        self.support = X[indices]  #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        print("the support is ",self.support.shape)
        self.b = y[indices[0]] - K[indices[0]]@(self.alpha*y) #''' -----------------offset of the classifier------------------ '''
        self.b = np.mean(y[indices] - K[indices]@(self.alpha*y)) #''' -----------------offset of the classifier------------------ '''

        beta = y*self.alpha
        self.norm_f = beta.reshape((1,-1))@K@beta  # '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x, indexes_of_change_size_subgraphs):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        output = (self.alpha*self.y).reshape((1,-1))@self.kernel(self.X,x, indexes_of_change_size_subgraphs)
        return output.flatten()
    
    
    def predict(self, X, indexes_of_change_size_subgraphs):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X, indexes_of_change_size_subgraphs)
        return 2 * (d+self.b) - 1


if __name__ == "__main__":

    sigma = 1. # this parameter is useless for Min-Max kernel
    C= 100.
    kernel = RBF(sigma).kernel
    model = KernelSVC(C=C, kernel=kernel)
    max_subgraph_size = 8

    # we load allready vectorized data. please use 'vectorization_saver.ipynb' to vectorize the data yourself

    X_training = load_npz('./vectorizations_ordered/X_train_max_subgraph_' + str(max_subgraph_size) + '.npz')
    training_labels = np.load('./training_labels.pkl', allow_pickle=True)
    X_test = load_npz('./vectorizations_ordered/X_test_max_subgraph_' + str(max_subgraph_size) + '.npz')
    indexes_of_change_size_subgraphs = np.load("./vectorizations_ordered/indexes_of_change_size_subgraphs_" + str(max_subgraph_size) + ".npy")
    model.fit(X_training,training_labels, indexes_of_change_size_subgraphs)
    estimated_test_labels = np.array(model.predict(X_test,indexes_of_change_size_subgraphs)).flatten()
    Yte = {'Predicted' : np.round(np.array(estimated_test_labels).flatten(),3)} 
    df = pd.DataFrame(Yte) 
    df.index += 1 
    df.to_csv('test_pred.csv',index_label='Id')

    print("done ! The result is in test_pred.csv !")













from numpy import array, dot
from qpsolvers import solve_qp
import bayes_opt
from bayes_opt import BayesianOptimization
#from Strategy import errorLoss, rollingWindowsValidation

#Funcion para Trasposición conjugada compleja en Python
from numpy import ndarray
class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T

from Strategy import getWeights, rollingWindowsValidation
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles

class EW:
    def __init__(self, data):
        self.data = []
    pass
    
    #The Equally Weighted Minimum Variance approach
        #   This class derives from the Strategy Class and implements the
        #   optimization problem associated to the Markowitz's theory with
        #   explicit diversification in the cost function 

    #Description: Relative importance of the variance.
    
    class obj:
        name = 'Equally Weighted Strategy'
        pass
    
    
    # Description: This function runs the corresponding strategy, fitting the model weights. 
    def solveOptimizationProblem(obj, data, vars):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 
        (numElements, N) = data.shape
        # mean and covariance
        
        W = np.ones((N,1))*(1/N)
        
        return W
    
    def config(obj,data,vars, varsCV):
        
        return obj     








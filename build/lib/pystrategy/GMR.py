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

class GMR:
    def __init__(self, data):
        self.data = []
    pass
    
    #The Equally Weighted Minimum Variance approach
        #   This class derives from the Strategy Class and implements the
        #   optimization problem associated to the Markowitz's theory with
        #   explicit diversification in the cost function 

    #Description: Relative importance of the variance.
    
    class obj:
        name = 'Global Maximum Return Strategy'
        pass
    
    
    # Description: This function runs the corresponding strategy, fitting the model weights. 
    def solveOptimizationProblem(obj, data, vars):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 
        (numElements, N) = data.shape
        # Compute the mean return
        #meanReturn  =  
        meanReturn=np.asarray(np.cumsum((np.mean(data, axis=0)), axis=0))[0]
        #Global maximun return approach
        ValueMax = meanReturn.max()
        indexMax=np.where(meanReturn==meanReturn.max())[0][0]
        W = np.zeros((N,1))
        W[[indexMax]]=1
        
        return W
    
    def config(obj,data,vars, varsCV):
            
        return obj     

    







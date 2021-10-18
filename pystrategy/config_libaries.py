from abc import abstractmethod
import pandas as pd
import numpy as np
from zipfile import *
from Strategy import *
from classes_strategies.Optimizers import *
from classes_strategies.Strategies import *
from MaxDrawdown import *
from classes_strategies.Factory import *
from numpy import array, dot


"""
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
"""
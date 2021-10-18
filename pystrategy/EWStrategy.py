from EW import EW
from MaxDrawdown import MaxDrawdown 
from Strategy import getWeights, rollingWindowsValidation
#from Strategy import errorLoss, rollingWindowsValidation
import pandas as pd
import numpy as np
#Funcion para Trasposición conjugada compleja en Python
from numpy import ndarray
class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T




ewStrategy = EW(data)

class vars:
    pass

class varsCV:
    pass

vars.validationWindows = 36
vars.CVWindows = 12

ewStrategy.config(data,vars, varsCV)
returns = rollingWindowsValidation(ewStrategy,data,vars)

w = getWeights(ewStrategy)

[MDD, MDDs, MDDe, MDDr] = MaxDrawdown(returns)

MR = returns.mean()
SR = MR/np.std(returns)

CR = MR/MDD

(Q, N) = w.shape
Turnover = (1/(Q-1))*(1/N)* sum(sum(abs(w[2:,:]-w[1:-1,:])))

print('MR: {}, SR: {}, CR:{}, Turnover: {}'.format(MR,SR,CR,Turnover))





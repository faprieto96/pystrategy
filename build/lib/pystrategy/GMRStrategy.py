from GMR import GMR
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


main_path='/Users/franciscoantonioprietorodriguez/Documents/PhD/GitHub_Code_Repository/PhD/'
data_path=main_path+'data/6_Emerging_Markets_8years.csv'

data= np.matrix(pd.read_csv(data_path, header=None))

mvStrategy = GMR(data)

class vars:
    pass


vars.validationWindows = 36
vars.CVWindows = 12

mvStrategy.config(mvStrategy.obj, data,vars)
returns = rollingWindowsValidation(mvStrategy,data,vars)

w = getWeights(mvStrategy)

[MDD, MDDs, MDDe, MDDr] = MaxDrawdown(returns)

MR = returns.mean()
SR = MR/np.std(returns)

CR = MR/MDD

(Q, N) = w.shape
Turnover = (1/(Q-1))*(1/N)* sum(sum(abs(w[2:,:]-w[1:-1,:])))

print('MR: {}, SR: {}, CR:{}, Turnover: {}'.format(MR,SR,CR,Turnover))





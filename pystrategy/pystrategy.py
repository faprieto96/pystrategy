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




main_path='/Users/franciscoantonioprietorodriguez/Documents/PhD/GitHub_Code_Repository/PhD/'
data_path=main_path+'data/6_Emerging_Markets_8years.csv'

data= np.matrix(pd.read_csv(data_path, header=None))


class vars:
    pass

class varsCV:
    pass



class Strategy(vars, varsCV):

    def __init__(self, data, strategy, validationWindows, CVWindows):
        self.data=data
        self.strategy=strategy
        self.validationWindows=validationWindows
        self.CVWindows=CVWindows
        
    
    vars.validationWindows = 12
    vars.CVWindows = 36
    
    def run(self):
        results={}
        if self.strategy=='EW':
            strategy_selected=EW(self.data)
            strategy_selected.config(self.data,vars, varsCV)
            returns = rollingWindowsValidation(strategy_selected,self.data,vars)
            results['returns'] = rollingWindowsValidation(strategy_selected,self.data,vars)
            results['weigths'] = getWeights(strategy_selected)
            w = getWeights(strategy_selected)

            [MDD, MDDs, MDDe, MDDr] = MaxDrawdown(returns)

            MR = returns.mean()
            results['MR'] = MR
            SR = MR/np.std(returns)
            results['SR'] = SR
            CR = MR/MDD
            results['CR'] = CR
            (Q, N) = w.shape
            Turnover = (1/(Q-1))*(1/N)* sum(sum(abs(w[2:,:]-w[1:-1,:])))
            results['Turnover'] = Turnover
            return results
        elif self.strategy=='GMR':
            return EW(data)

print('prueba')







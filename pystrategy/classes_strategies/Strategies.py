
from abc import abstractmethod
from classes_strategies.Optimizers import *
from config_libaries import *



class Str:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    @abstractmethod
    def run(self, opt, *args, **kwargs):
        ...


class Str1(Str):
    def run(self, opt: Opt, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        print('Optimizer:', opt.__class__)
        var = 'variable generada en RUN Str1'
        r = opt()(var, args, kwargs)

        return r


class Str2(Str):
    def run(self, opt: Opt, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        print('Optimizer:', opt.__class__)
        var = 'variable generada en RUN Str2'
        r = opt()(var, args, kwargs)

        return r

class Str8(Str):
    def run(self, opt: Opt, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        print('Optimizer:', opt.__class__)
        var = 'variable generada en RUN Str8'  + " " +str(kwargs['beta']) + " " +str(kwargs['lambda'])
        r = opt()(var, args, kwargs)

        return r

class EW(Str):
    def inizialization(self, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        #print('Optimizer:', opt.__class__)
        
        name = 'Equally Weighted Strategy'
        __inizialization={'name':name}
        return __inizialization
        
    def solveOptimizationProblem(self, *args, **kwargs):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 
        
        (numElements, N) = args[0]['intermediate_data'].shape
        # mean and covariance
        
        W = np.ones((N,1))*(1/N)
        
        return W


class GMR(Str):
    
    def inizialization(self, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        #print('Optimizer:', opt.__class__)
        
        name = 'Global Maximum Return Strategy'
        __inizialization={'name':name}
        return __inizialization
        
    def solveOptimizationProblem(self, *args, **kwargs):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 
        (numElements, N) = args[0]['intermediate_data'].shape
        # Compute the mean return
        meanReturn=np.asarray(np.cumsum((np.mean(args[0]['intermediate_data'], axis=0)), axis=0))[0]
        #Global maximun return approach
        ValueMax = meanReturn.max()
        indexMax=np.where(meanReturn==meanReturn.max())[0][0]
        W = np.zeros((N,1))
        W[[indexMax]]=1
        return W
# Rellenar con class <estrategias>

class StrTypes:
    STR1 = 'str1'
    STR2 = 'str2'
    STR8 = 'str8'
    EW = 'EW'
    GMR = 'GMR'
    # Rellenar con tipos STR_N estrategias

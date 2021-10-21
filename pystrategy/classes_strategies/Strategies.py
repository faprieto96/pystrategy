
from abc import abstractmethod
from config_libaries import *
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles
from qpsolvers import solve_qp
import bayes_opt
from bayes_opt import BayesianOptimization
from numpy import ndarray
class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T
import numpy as np
from numpy import array, dot
from scipy.optimize import basinhopping


"""class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T"""




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
    def run(self, str: Opt, *args, **kwargs):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        return s_object.inizialization()

        


##############################################################################################
# EQUALLY WEIGHTED APPROACHES:
#   EQUALLY WEIGHTED (EW)
#   EQUALLY WEIGHTED RISK CONTRIBUTION (EWRC)
#   MAXIMUM DIVERSIFICATION (MD)
##########################################################################
##########################################################################
##################                       ################################
##################       --------        ###############################
##################         EW            ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################                       ##########################
#################################################################
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

##########################################################################
##################                       ################################
##################       ----!----        ###############################
##################         EWRC          ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################  NECESITO ENCONTRAR EQUIVALENTE FMINCON ##########################
#################################################################
class EWRC(Str):
    
    def inizialization(self, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        #print('Optimizer:', opt.__class__)
        
        name = 'Equally Weighted Risk Contribution Strategy'
        __inizialization={'name':name}
        return __inizialization
    
    
    
        
    def solveOptimizationProblem(self, *args, **kwargs):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 
        
        (numElements, N) = args[0]['intermediate_data'].shape
        # mean and covariance
        Sigma   = EmpiricalCovariance().fit(args[0]['intermediate_data']).covariance_*12              # I use 12 for annualizing the covmatrix
        
        def func(w, Omega):
            x = 0
            R = Omega*w
            for i in range(1,len(w)):
                for j in range(1,len(w)):
                    x = x + (w(i)*R(i)-w(j)*R(j))**2
            x = x/(w*R)
            return x
        
        
        # Sequential Quadratic Programming
        #f = @(w) obj.func(w,Sigma)
        #func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
        #x0=[0.5]

        #minimizer_kwargs = {"method": "BFGS"}
        #ret = basinhopping(, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
        A = np.ones((1,N))
        b = 1
        lb = np.zeros((N,1))
        w0=1/N*np.ones((N,1))

        #opts    = optimset('Display','off')

        W = fmincon((lambda w: func(w, Sigma)),w0,[],[],A,b,lb,[],[],opts)
        
        return W

##########################################################################
##################                       ################################
##################       ----!----        ###############################
##################         MD            ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################                       ##########################
#################################################################
class MD(Str):
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


##############################################################################################
# MARKOWITZ BASELINE FORMULATION:
#   GLOBAL MAXIMUM RETURN (GMR)
#   GLOBAL MINIMUM VARIANCE (GMV)
##########################################################################
##########################################################################
##################                       ################################
##################       ----!----        ###############################
##################         GMR           ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################                       ##########################
#################################################################
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

##########################################################################
##################                       ################################
##################       ----!----        ###############################
##################         GMV           ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################                       ##########################
#################################################################
class GMV(Str):
    
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
        
        # mean and covariance
        Sigma   = np.cov(args[0]['intermediate_data'], rowvar=False)*12          # I use 12 for annualizing the covmatrix
                                          # mean log returns
        
        H = 2*Sigma
        f = [] #FALTA TRANSPOSE

        Aeq     = np.ones((1,N))
        beq     = 1
        LB      = np.zeros((1,N))                                         
        UB      = np.ones((1,N))                                                     
        #opts    = optimset('Algorithm', 'interior-point-convex', 'Display','off')
        #   Revisar cómo meter la opción de 'interior-point-convex'

        # Python reference for quadprog: 
        #   https://pypi.org/project/qpsolvers/
        #Original funct (it contains opts) (Wa, varP)  = solve_qp(H,f,[],[],Aeq,beq,LB,UB,UB/N,opts) 
        
        P=H
        try:
            q=np.asarray(f).reshape((6,))
        except:
            q=np.zeros((1,N)).reshape((6,))
        G=np.zeros((6,6))
        h=np.zeros(6)
        A=np.asarray(Aeq).reshape((6,))
        b=np.array([beq])
        lb=LB
        ub=UB
        
        """M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
        P = dot(M.T, M)  # this is a positive definite matrix
        q = dot(array([3., 2., 3.]), M).reshape((3,))
        G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        h = array([3., 2., -2.]).reshape((3,))
        A = array([1., 1., 1.])
        b = array([1.])"""

        #(Wa, varP, third_parameter) = solve_qp(P, q, G, h, A, b)



        #(Wa, varP, third_parameter,fourth_parameter,fifth_parameter,sixt_parameter)  = solve_qp(P, q, G, h, A, b)
        W=np.array(solve_qp(P, q, G, h, A, b)) 
        #W = deltaValue* Wa + (1-deltaValue)*(1/N)*np.ones((N,1))
        return W


##############################################################################################
# APPROACHES THAT CONTROL THE UPPER AND LOWER BOUNDS:
#   WEIGTH UPPER-BOUND CONSTRAINT (WUBC)
#   WEIGTH LOWER-BOUND CONSTRAINT (WUBC)
##########################################################################
##########################################################################
##################                       ################################
##################       ----!----        ###############################
##################         WUBC           ##############################
##################       STRATEGY        #############################
##################       --------        ############################
##################                       ##########################
#################################################################
class WUBC(Str):
    
    def inizialization(self, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        #print('Optimizer:', opt.__class__)
        
        name = 'Weight Upper-Bound Constraint'
        __inizialization={'name':name}
        return __inizialization
    
    def config(self, *args, **kwargs):
        (variable,N) = args[0]['intermediate_data'].shape
        
        # Create the validation set
        dataValidation = data[0:-args[0]['validationWindows']:,:]
        # Compute the CV windows
        validationWindows = args[0]['CVWindows']
        # Create the optimization variable
        
        sexp = squaredExponential()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')
        param = {'lambda_value': ('cont', [0,1]), 'upperBound': ('cont', [0,1])}
        #param es equivalente en MATLAB a:
        #   num = WUBC.obj.lambda_value
        #   ub = WUBC.obj.upperBound
        #Bayesian optimization:
        def rollingWindowsValidation(dataValidation, str_params: dict):
            # Save number of elements and number of assets
            (numData, N) = dataValidation.shape
            # Initialize the weights matrix
            W = np.zeros((str_params['vars_validationWindows'], N))
            
            for i in range(0,(str_params['vars_validationWindows'])):
                str_params['intermediate_data']=str_params['data'][i:numData-str_params['vars_validationWindows']+(i),:]
                W[[i]] = np.transpose(s_object.solveOptimizationProblem(str_params))
                
            str_params['W']=W
            a=W
            b=(str_params['data'][str_params['data'].shape[0] - str_params['vars_validationWindows']:,:])
            #return np.multiply(x1, x2) obj.w * (data[data.shape[0] - vars.validationWindows:,:]).mean(axis=0)
            #(sum(OK'))';
            return np.multiply(a,b).sum(axis=1, dtype='float')

        def errorLoss(lambda_value, upperBound):
            vars.lambda_value = lambda_value
            vars.upperBound = upperBound
            returns = rollingWindowsValidation(dataValidation, varsCV)
            value = np.std(returns)/returns.mean()
            return value
            
        my_func = lambda lambda_value, upperBound : errorLoss(lambda_value, upperBound)
        results = GPGO(gp, acq, errorLoss, param)
        results.run(max_iter=10)
        #Extract the best parameters
        #obj.lambda_value = results.best[[0]]
        #obj.upperBound = results.best[[1]]
        obj.lambda_value =results.getResult()[0]['lambda_value']
        obj.upperBound =results.getResult()[0]['upperBound']
        #obj.lambda_value = 0.99
        #obj.upperBound = 0.3128
        
        return obj  
        
    def solveOptimizationProblem(self, *args, **kwargs):
        # Type: It returns the optimized weights
        # Compute numbers of data points and assets 

        (numElements, N) = args[0]['intermediate_data'].shape
        Sigma   = np.cov(args[0]['intermediate_data'], rowvar=False)*12       # I use 12 for annualizing the covmatrix
        Vars    = np.diag(Sigma)                                              # variances of the stocks
        mu      = args[0]['intermediate_data'].mean(axis=0).H*12                                      # mean log returns
            
        if False==hasattr(vars,'lambda_value'):
            # third parameter does not exist, so default it to something
            lambdaValue = WUBC.obj.lambda_value
        else:
            lambdaValue = vars.lambda_value
        
        
        if False==hasattr(vars,'upperBound'):
            # third parameter does not exist, so default it to something
            upperBoundValue = WUBC.obj.upperBound
        else:
            upperBoundValue = vars.upperBound
        
        
        H = 2*(lambdaValue*Sigma)
        f = - mu.H

        Aeq     = np.ones((1,N))
        beq     = 1
        LB      = np.zeros((1,N))                                       
        UB      = np.ones((1,N))*upperBoundValue                                                       
        #opts    = optimset('Algorithm', 'interior-point-convex', 'Display','off')
        #   Revisar cómo meter la opción de 'interior-point-convex'

        # Python reference for quadprog: 
        #   https://pypi.org/project/qpsolvers/
        #Original funct (it contains opts) (Wa, varP)  = solve_qp(H,f,[],[],Aeq,beq,LB,UB,UB/N,opts) 
        
        P=H
        q=np.asarray(f).reshape((6,))
        G=np.zeros((6,6))
        h=np.zeros(6)
        A=np.asarray(Aeq).reshape((6,))
        b=np.array([beq])
        lb=LB
        ub=UB
        
        from numpy import array, dot
        from qpsolvers import solve_qp

        """M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
        P = dot(M.T, M)  # this is a positive definite matrix
        q = dot(array([3., 2., 3.]), M).reshape((3,))
        G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        h = array([3., 2., -2.]).reshape((3,))
        A = array([1., 1., 1.])
        b = array([1.])"""

        #(Wa, varP, third_parameter) = solve_qp(P, q, G, h, A, b)



        W=np.array(solve_qp(P, q, G, h, A, b)) 
        return W


class StrTypes:
    STR1 = 'str1'
    EW = 'EW'
    EWRC = 'EWRC'
    MD = 'MD'
    GMR = 'GMR'
    GMV = 'GMV'
    WUBC = 'WUBC'

    # Rellenar con tipos STR_N estrategias

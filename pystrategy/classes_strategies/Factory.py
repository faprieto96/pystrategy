
from config_libaries import *

class StrFactory:
    strategies = {
        StrTypes.STR1: Str1,
        StrTypes.STR2: Str2,
        StrTypes.STR8: Str8,
        StrTypes.EW: EW,
        StrTypes.GMR: GMR,
        # Rellenar con tipos estrategias
    }

    #Metodo para obtener el nombre de la estrategia seleccionada
    @staticmethod
    def inizialization(name,str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        return s_object.inizialization()


    @staticmethod
    def rollingWindowsValidation(name, str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        # Save number of elements and number of assets
        (numData, N) = str_params['data'].shape
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


    #Metodo para obtener solveOptimizationProblem
    @staticmethod
    def solveOptimizationProblem(name, str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        str_params['W'] = s_object.solveOptimizationProblem(str_params)
        return str_params
    
    """#Metodo build corre la estrategia que se desee
    @staticmethod
    def build(name, opt: Opt, str_params: dict, opt_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        return s_object.run(opt, **opt_params)"""

    @staticmethod
    def build(name, str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        return s_object.run(opt, **opt_params)
    

    


    @staticmethod
    def MaxDrawdown(name, str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        # Save number of elements and number of assets
        n = max(str_params['returns'].shape)
        # calculate vector of cum returns
        cr = np.cumsum((np.asarray(str_params['returns'])).flatten(), axis=0)
        # calculate drawdown vector
        dd=[]
        for i in range(1,n):
            dd.append(max(cr[0:i])-cr[i-1])
        dd=np.array(dd)
        # calculate maximum drawdown statistics
        MDD = max(dd)
        MDDe = np.where(dd==MDD)[0][0]
        try:
            MDDs = np.where(abs(cr[MDDe]+ MDD - cr) < 0.000001)[0][0]
        except:
            MDDs=0
        try:
            MDDr = np.where(MDDe+min(cr[MDDe:] >= cr[MDDs]))[0]-1
        except:
            try:
                MDDr = np.where(MDDe+min(cr>= cr[MDDs]))[0]-1
            except:
                MDDr = []
        str_params['MDD']=MDD
        str_params['MDDs']=MDDs
        str_params['MDDe']=MDDe
        str_params['MDDr']=MDDr   
        return str_params

    @staticmethod
    def Output_financial_ratios(name, str_params: dict):
        s_name = StrFactory.strategies.get(name)
        s_object = s_name(**str_params)
        # Save number of elements and number of assets
        n = max(str_params['returns'].shape)
        MR = str_params['returns'].mean()
        SR = MR/np.std(str_params['returns'])

        CR = MR/str_params['MDD']

        (Q, N) = str_params['W'].shape
        Turnover = (1/(Q-1))*(1/N)* sum(sum(abs(str_params['W'][2:,:]-str_params['W'][1:-1,:])))
        str_params['MR']=MR
        str_params['SR']=SR
        str_params['CR']=CR
        str_params['Turnover']=Turnover  

        print('MR: {}, SR: {}, CR:{}, Turnover: {}'.format(MR,SR,CR,Turnover))
        return str_params
    

    
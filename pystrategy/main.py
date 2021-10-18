from config_libaries import *




main_path='/Users/franciscoantonioprietorodriguez/Documents/PhD/GitHub_Code_Repository/PhD/'
data_path=main_path+'data/6_Emerging_Markets_8years.csv'

data= np.matrix(pd.read_csv(data_path, header=None))




#Generamos la funcion py_strategy que llama a la factoria y la función
def py_strategy(str_name, optimizer, str_params, opt_params):
    """
    PY STRATEGY
    :param str_name:
    :param optimizer:
    :param str_params:
    :param opt_params:
    :return:
    """
    #Obtención del nombre de la estrategia
    __inizialization=StrFactory.inizialization(str_name,str_params)
    __inizialization={'initial_parameters':__inizialization}

    #Se corre el rollingWindowsValidation
    returns= StrFactory.rollingWindowsValidation(str_name, str_params)
    str_params['returns']=returns
    returns=StrFactory.MaxDrawdown(str_name, str_params)
    
    return str_params

_str_params = {'A': 1, 'B': 2, 'data':data, 'vars_validationWindows': 36, 'vars_CVWindows':12}
_opt_params = {'lambda': 0.2, 'beta': 0.1}


res = py_strategy(StrTypes.EW, OptTypes.BAYES, _str_params, _opt_params)
print('Output:', res)

# class AStrategy(Strategy):
#
#     def run(self, *args, **kwargs):
#         # 1- Params checking.
#         print(f'{args}: {kwargs}')  # Press Ctrl+F8 to toggle the breakpoint.
#         return kwargs.get('lamb') + kwargs.get('beta')
#
#
# class PyStrategy:
#
#     @staticmethod
#     def run(f, *args, **kwargs):
#         print('Before')
#         r = f(*args, **kwargs)
#         print('After:', r)
#
#
# PyStrategy.run(estrategia_b, 'a', 'b', lamb=0.1, beta=0.2)





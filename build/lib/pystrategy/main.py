from config_libaries import *

main_path='/Users/franciscoantonioprietorodriguez/Documents/PhD/GitHub_Code_Repository/PhD/'
data_path=main_path+'data/6_Emerging_Markets_8years.csv'
data= np.matrix(pd.read_csv(data_path, header=None))


#Generamos la funcion py_strategy que llama a la factoria y la función
def py_strategy(str_name, optimizer, str_params, opt_params):
    """ La documentación devuelve la siguiente información
    PY STRATEGY
    
    Parameters
    ----------
    :param str_name: Estrategia a seleccionar
    :param optimizer: Optimizador a seleccionar
    :param str_params: Parametros de entrada para el modelo
    :param opt_params: Definición de los hiperparámetros

    Return
    ------
    :return: Información de los ratios financieros
    """

    # Obtención del nombre de la estrategia y/o parametros iniciales
    __inizialization=StrFactory.inizialization(str_name,str_params)
    __inizialization={'initial_parameters':__inizialization}

    # Se corre el rollingWindowsValidation
    returns= StrFactory.rollingWindowsValidation(str_name, str_params)
    str_params['returns']=returns

    # MaxDrawdown function
    returns=StrFactory.MaxDrawdown(str_name, str_params)

    # Financial ratios function
    str_params=StrFactory.Output_financial_ratios(str_name, str_params)
    
    #Return of final function
    return str_params #En return de la función puedo seguir incluyendo diferentes parámetros
    

#Data by default
_str_params = {'data':data, 'vars_validationWindows': 36, 'vars_CVWindows':12}
_opt_params = {'lambda': 0.2, 'beta': 0.1}


res = py_strategy(StrTypes.GMR, OptTypes.BAYES, _str_params, _opt_params)
print('Output:', res)


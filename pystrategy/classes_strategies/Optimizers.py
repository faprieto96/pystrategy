
from config_libaries import *

class Opt:
    @abstractmethod
    def run(*args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class BayesOpt(Opt):
    def run(*args, **kwargs):
        # Rellenar
        output_optimization={'val_lambda':args[2]['lambda'], 'val_beta':args[2]['beta']}
        Output_strategy_parameters=args[3]
        #return 'bayes -> ({}), {}, {}'.format(args[0], args[1:], kwargs, output_optimization)
        return output_optimization, Output_strategy_parameters


class NestedOpt(Opt):
    def run(*args, **kwargs):
        #args[3] devuelve los parámetros
        # Rellenar

        return 'PRUEBA!!! -> ({}), {}, {}'.format(args[0], args[1:], kwargs)


class OptTypes:
    BAYES = BayesOpt
    NESTED = NestedOpt
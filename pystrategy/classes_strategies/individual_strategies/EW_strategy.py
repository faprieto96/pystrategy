from abc import abstractmethod
from classes_strategies.Optimizers import *

class EW(Str):
    def run(self, opt: Opt, *args, **kwargs):
        print('Params class:', self.args, self.kwargs)
        print('Params Strategy Run:', args, kwargs)
        print('Optimizer:', opt.__class__)

        var = 'STRATEGY EW '  + " " +str(kwargs['beta']) + " " +str(kwargs['lambda'])
        r = opt()(var, args, kwargs)

        return r
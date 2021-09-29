import numpy as np
from zipfile import *


def rollingWindowsValidation(obj, data, vars):
    # Save number of elements and number of assets
    (numData, N) = data.shape
    
    # Initialize the weights matrix
    W = np.zeros((vars.validationWindows, N))
    
    for i in range(0,(vars.validationWindows)):
        W[[i]] = np.transpose(obj.solveOptimizationProblem(data[i:numData-vars.validationWindows+(i),:],vars))
        
    
    obj.w = W
    a=obj.w
    b=(data[data.shape[0] - vars.validationWindows:,:])
    #return np.multiply(x1, x2) obj.w * (data[data.shape[0] - vars.validationWindows:,:]).mean(axis=0)
    #(sum(OK'))';
    return np.multiply(a,b).sum(axis=1, dtype='float')

def getWeights(obj):
    weights = obj.w
    return weights


"""classdef Strategy < handle
    % Strategy Abstract interface class
    % Abstract class which defines portfolio-related strategies.
    % It describes some common methods and variables for all the
    % strategies.
    
    properties
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Variable: w (Public)
        % Type: Vector
        % Description: Vector of weights associated to the
        %              strategy
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        w = []
        name
        
    end
    
 
    methods
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: setWeights (Public)
        % Description: It assigns the weights to the strategy.
        % Type: Void
        % Arguments:
        %           value--> Vector of weights.
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = setWeights(obj, value)
                obj.w = value;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: getWeights (Public)
        % Description: It returns the weights of the implemented
        %               strategy.
        % Type: Vector
        % Arguments:
        %           No arguments
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function weights = getWeights(obj)
            weights = obj.w;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: getName (Public)
        % Description: It returns the name of the implemented
        %               strategy.
        % Type: String
        % Arguments:
        %           No arguments
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function name = getName(obj)
            name = obj.name;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: setName (Public)
        % Description: It assigns the name to the strategy.
        % Type: Void
        % Arguments:
        %           value--> name of the strategy.
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function obj = setName(obj, value)
                obj.name = value;
        end
        
        function returns = rollingWindowsValidation(obj, data, vars)
            
            % Save number of elements and number of assets
            [numData N] = size(data);
            
            % Initialize the weights matrix
            W = zeros(vars.validationWindows, N);
            
            for i =1:1:vars.validationWindows
                W(i,:) = solveOptimizationProblem(obj,data(i:numData-vars.validationWindows+(i-1),:),vars);
            end
            
            obj.w = W;
            
            returns = (sum((obj.w .* data(size(data,1)-vars.validationWindows+1:end,:))'))';
        
        end

    end
    
    
    methods(Abstract)
        %%%%%%%%%%%%%%%%%%%%%%%%-%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: solveOptimizationProblem(Public)
        % Description: This function is not implemented,
        %                   since it will be designed by
        %                   the derivated class.
        % Type: Void
        % Arguments:
        %           No arguments
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        solveOptimizationProblem(obj, data, vars);
        
        %%%%%%%%%%%%%%%%%%%%%%%%-%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % Function: config(Public)
        % Description: This function is not implemented,
        %                   since it will be designed by
        %                   the derivated class.
        % Type: Void
        % Arguments:
        %           No arguments
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        config(obj, data, vars);
        
        
        
    end
        
end"""
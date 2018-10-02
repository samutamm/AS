
from Module import Module

class Function(Module):
    
    @staticmethod
    def forward(W, X):
        return X.t().mm(W)
    
    @staticmethod
    def backward(grad_output, input):
        #
        #
        return None
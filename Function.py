from Module import Module
    
class Function(Module):

    @staticmethod
    def forward(W, X):
        return W.mm(X)
    
    @staticmethod
    def backward(W, dZ, y_h):
        
        dW = dZ.t().mm(y_h.t()).t()
        dA = W.t().mm(dZ.t())
                
        return dA, dW
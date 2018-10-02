
import torch

class Loss:
    
    def forward(self, y, ypred):
        pass
    
    def backward(self, y, ypred):
        pass
    
class MSE(Loss):
    
    def forward(self, y, ypred):
        return 1/2 * ((y - ypred) ** 2).mean()
    
    def backward(self, y, ypred):
        return ypred - y
    
class Hinge(Loss):
    
    def forward(self, y, ypred):
        m = ypred.shape[0]
        v = y.view(m, 1).float() * ypred.float()

        return torch.max(torch.zeros(1), -v)       
        
    def backward(self, y, ypred):
        return torch.where(y*ypred < 1, -1 * y, torch.zeros(1))
        #m = ypred.shape[0]
        #v = y.float() * ypred.float()
        #grad = -y
        #grad[v >= 1] = 0
        #return grad
        
class CrossEntropy(Loss):
    """
        ATTENTION: For multiclass and this one, must use
        softmax to convert ypred to probability.
    """
    
    def forward(self, y, yhat):
        return torch.sum(y * torch.log(yhat) + (1 - y)*(torch.log(1 - yhat)))
    
    def backward(self, y, ypred):
        n = ypred.shape[0]
        return 1/n * (ypred - y)
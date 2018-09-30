
import torch

class Module:
   
    def backward_update_gradient(self,X,delta):
        m = X.shape[1]
        self.gradient_W = 1 / m * torch.mm(X,delta.t())
        #
    
    def update_parameters(self, epsilon):
        self.W -= epsilon * self.gradient_W
    
    def grad_zero(self):
        self.gradient_W = torch.zero_(self.gradient_W)
        
 #   def predict(self, X):
 #       """ 
 #         Helps testing.
 #       """
 #       return self.forward(torch.from_numpy(X.reshape(-1,X.shape[0])).float())
    
class Lineaire(Module):
    
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def initialize_parameters(self):
        self.W = torch.FloatTensor(self.in_dim, self.out_dim)
        self.W = torch.nn.init.xavier_uniform_(self.W)
        self.gradient_W = torch.FloatTensor(self.in_dim, self.out_dim)
        self.grad_zero()
        
    def forward(self, X):
        return self.W.t() @ X
    
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
    
class CrossEntropy(Loss):
    """
        ATTENTION: For multiclass and this one, must use
        softmax to convert ypred to probability.
    """
    
    def forward(self, y, ypred):
        return torch.sum(y * torch.log(yhat) + (1 - y)*(torch.log(1 - yhat)))
    
    def backward(self, y, ypred):
        n = ypred.shape[0]
        return 1/n * (ypred - y)
    
    
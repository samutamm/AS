
import torch

class Module:
   
    def backward_update_gradient(self,X,delta):
        #m = X.shape[1]
        self.gradient_W += X.t().mm(delta)
        #self.gradient_W += torch.mm(X,delta.t())
        # 1 / m
    
    def update_parameters(self, epsilon):
        self.W -= epsilon * self.gradient_W.t()
    
    def grad_zero(self):
        self.gradient_W = torch.zero_(self.gradient_W)
        
    
class Lineaire(Module):
    
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def initialize_parameters(self):
        self.W = torch.rand(self.out_dim, self.in_dim)
        #self.W = torch.FloatTensor(self.in_dim, self.out_dim)
        #self.W = torch.nn.init.xavier_uniform_(self.W)
        self.gradient_W = torch.zeros(self.in_dim, self.out_dim)
        self.grad_zero()
        
    def forward(self, X):
        return X.matmul(self.W.t())
    
    #return X.matmul(self.w.t())
    

    
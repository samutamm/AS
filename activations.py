import torch

from Module import Module

class SigmoidActivation(Module):
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    def sigmoid_prime(self, z):
        s = self.sigmoid(z)
        return s*(1-s)
    
    def forward(self, z):
        """ 
            Activation function = sigmoid
        """
        return self.sigmoid(z)
    
    def backward(self, dA, Z):
        dZ = dA * self.sigmoid_prime(Z)

        assert (dZ.shape == Z.shape)
        
        return dZ


class ReluActivation(Module):
    
    def forward(self, Z):
        return torch.max(torch.zeros(1), Z)
    
    def backward(self, Z):
        dZ = Z.clone()
        dZ[Z <= 0] = 0
        return dZ
    

class TanhActivation(Module):
    
    def forward(self, Z):
        return (torch.exp(Z) - torch.exp(-Z)) / (torch.exp(Z) + torch.exp(-Z))
    
    def backward(self, dA, Z):
        derivative = (1 - self.forward(Z) ** 2)
        #import pdb; pdb.set_trace()
        dZ = dA * derivative
        #dZ = dZ.t()
        
        assert (dZ.shape == Z.shape)
        
        return dZ
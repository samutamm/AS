import torch

from Module import Module

class SigmoidActivation(Module):
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def forward(self, z):
        """ 
            Activation function = sigmoid
        """
        return 1 / (1 + torch.exp(-z))
    
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        sp = self.sigmoid_prime(input)
        return delta * sp


class ReluActivation(Module):
    
    def forward(self, Z):
        return torch.max(torch.zeros(1), Z)
    
    def backward(self, Z):
        dZ = Z.clone()
        dZ[Z <= 0] = 0
        return dZ
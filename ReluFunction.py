
from torch.autograd import Function
from torch.nn.modules import Module

class ReluFunction(Function):
    @staticmethod
    def forward(ctx, X):
        result = X.clamp(min=0)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[result <= 0] = 0
        return grad_x
    
class Relu(Module):
   
    def forward(self, input):
        return input.clamp(min=0)
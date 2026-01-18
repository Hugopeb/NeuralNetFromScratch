import torch

class Dense():
    def __init__(self, output_size, input_size);
        self.weights = torch.randn(output_size, input_size)
        self.bias = torch.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = self.input @ self.weights + self.bias
        return self.output

    def backwards(self, grad_output):
        return 
    

class ReLU():
    def forward(self, input):
        self.mask = input > 0.0
        return self.mask * input
    
    def backwards(self, grad_output):
        return grad_output * self.mask
    
    
class Tanh():
    def forward(self, input):
        output = torch.tanh(input)
        return output
    
    def backwards(self, grad_output):
        grad_input = 1 - torch.tanh(grad_output)**2
        return grad_input
    

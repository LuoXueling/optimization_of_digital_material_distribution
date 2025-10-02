import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import torch.nn.functional as F

torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class Visco_Param_Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, min_val=0.0, max_val=1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_size, hidden_size, bias=False))
            current_size = hidden_size
        self.output_layer = nn.Linear(current_size, output_size, bias=False)
        self.min_val = min_val
        self.max_val = max_val
        
        self.activation = nn.Softplus()  # Softplus activation function
        #self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 



    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.sigmoid(self.output_layer(x))
        x = x * (self.max_val - self.min_val) + self.min_val  # Scale to [min_val, max_val]
        return x



    def get_weights(self):
        weights_hidden = [self.layers[i].weight.detach().cpu().numpy() for i in range(len(self.layers))]
        avweights = [weights_hidden] + [self.output_layer.weight.detach().cpu().numpy()]
        return avweights




if __name__ == '__main__':  
    # Example usage
    input_size = 1
    output_size = 1
    hidden_layer_sizes = [5, 5,]  # Define the sizes of each hidden layer
    model_visco_param = Visco_Param_Network(input_size, output_size, hidden_layer_sizes, min_val=0.0, max_val=1.0).to(device)


        

'''
Input Convex Neural Networks and Sparse Neural Networks through L0 regularization
Contact: 
Prof. Nikolaos Bouklas, Sibley School of Mechanical and Aerospace Engineering 
Cornell University, NY 14850, USA
Email: nb589@cornell.edu
'''
import torch
import torch.nn as nn
import math
torch.manual_seed(2019)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BETA = 2 / 3
BETA = torch.tensor(BETA).to(device)
GAMMA = -0.1
GAMMA = torch.tensor(GAMMA).to(device)
ZETA = 1.1
ZETA = torch.tensor(ZETA).to(device)
EPSILON = 1e-6
EPSILON = torch.tensor(EPSILON).to(device)

class deterministic_L0_layer(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(deterministic_L0_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.eps = None
        self.weight = torch.nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        torch.nn.init.trunc_normal_(self.weight, mean = 0.0, std=1.0, a=-0.1, b=0.1)
        
        self.qz_log_alpha = torch.nn.Parameter(torch.Tensor(self.input_dim,self.output_dim))

        self.floatTensor = torch.FloatTensor   if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.qz_log_alpha.data.normal_(math.log(1 - 0.5) - math.log(0.5), 1e-2)

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.nn.functional.sigmoid((torch.log(u).to(device) - torch.log(1.0-u).to(device) + self.qz_log_alpha) / BETA)
        return y * (ZETA - GAMMA) + GAMMA

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        # print("Hello")
        if self.eps is None or not reuse_u:
            self.eps =  (EPSILON - (1.0 - EPSILON)) * torch.rand(shape).to(device) + 1.0 - EPSILON
            #self.eps = torch.rand(shape=shape, minval=EPSILON, maxval=1.0 - EPSILON)
        return self.eps



    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.input_dim, self.output_dim))
            z = self.quantile_concrete(eps)
            return torch.clip_(z, 0., 1.)
        else:   # Mean of the hard concrete distribution
            pi = torch.nn.functional.sigmoid((self.qz_log_alpha))
            return torch.clip_(pi * (ZETA - GAMMA) + GAMMA, min=0.0, max=1.0)


    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.nn.functional.sigmoid(self.qz_log_alpha)
        return torch.clip_(pi * (ZETA - GAMMA) + GAMMA, min=0.0, max=1.0)



    def sample_weights(self, reuse_u=False):
        

        z = self.quantile_concrete(self.sample_u((self.input_dim, self.output_dim), reuse_u=reuse_u))
        mask = torch.nn.functional.hardtanh(z, min_val=0., max_val=1.)
        y = self.weight * mask
        return y
    

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.weight * self.get_z_mean()


    def loss(self):
        """Regularization loss term"""
        return torch.sum(torch.nn.functional.sigmoid(self.qz_log_alpha - BETA * math.log(-GAMMA / ZETA)))

    def clamp_weights(self):
        self.weight.data = torch.clamp(self.weight.data, min=0.0)


    def forward(self, x, sample=True, reuse_u=False):
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        return h
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from L0_layer import deterministic_L0_layer
import sys
torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class PICNN_network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_number_layers_details, input_dim1, hidden_number_layers_details1):
        super(PICNN_network, self).__init__()
        self.input_dim = input_dim
        self.input_dim1 = input_dim1
        self.hidden_number_layers_details1 = hidden_number_layers_details1
        self.output_dim = output_dim
        self.hidden_number_layers_details = hidden_number_layers_details

        self.activation = nn.Softplus()
        
        self.nc1 = deterministic_L0_layer(self.input_dim1, self.hidden_number_layers_details1[0])
        self.nc2 = nn.ModuleList([deterministic_L0_layer(self.hidden_number_layers_details1[i], self.hidden_number_layers_details1[i+1]) for i in range(len(self.hidden_number_layers_details1)-1)])

        self.ncfc1 = deterministic_L0_layer(self.hidden_number_layers_details1[0], self.hidden_number_layers_details1[0])
        self.ncfc2 = nn.ModuleList([deterministic_L0_layer(self.hidden_number_layers_details1[i], self.hidden_number_layers_details1[i+1]) for i in range(len(self.hidden_number_layers_details1)-1)])

        self.fc1 = deterministic_L0_layer(self.hidden_number_layers_details1[0]+self.input_dim, self.hidden_number_layers_details[0])
        self.fc2 = nn.ModuleList([deterministic_L0_layer(self.hidden_number_layers_details1[i]+self.hidden_number_layers_details[i]+self.input_dim, self.hidden_number_layers_details[i+1]) for i in range(len(self.hidden_number_layers_details)-1)])
        
        self.fc_final = deterministic_L0_layer(self.hidden_number_layers_details1[-1]+self.hidden_number_layers_details[-1]+self.input_dim, self.output_dim)


    def forward(self, x_in, nc_c, epoch,train_model=True):
        
        if epoch == 0 and train_model:

            y1 = self.activation(self.nc1(nc_c))
            c1 = torch.cat((x_in,self.activation(self.ncfc1(y1))),1)
            x1 = self.activation(self.fc1(c1))


            for fc, nc, ncfc in zip(self.fc2,self.nc2,self.ncfc2):
                y_out = self.activation(nc(y1))
                c2 = torch.cat((x_in,x1,self.activation(ncfc(y_out))),1)
                x_out = self.activation(fc(c2)) 
                y1 = y_out # Steven, we move to next layer
                x1 = x_out # Steven, we move to next layer

            x = self.fc_final(torch.cat((x_in,x1,y1),1))
        else:


            y1 = self.activation(self.nc1(nc_c,sample=False))
            c1 = torch.cat((x_in,self.activation(self.ncfc1(y1,sample=False))),1)
            x1 = self.activation(self.fc1(c1,sample=False))

            for fc, nc, ncfc in zip(self.fc2,self.nc2,self.ncfc2):
                y_out = self.activation(nc(y1,sample=False))
                c2 = torch.cat((x_in,x1,self.activation(ncfc(y_out,sample=False))),1)
                x_out = self.activation(fc(c2,sample=False))
                y1 = y_out # Steven, we move to next layer
                x1 = x_out # Steven, we move to next layer

            x = self.fc_final(torch.cat((x_in,x1,y1),1),sample=False)
        return x
    

    def getLossRegL0(self):
        loss_fc = self.fc1.loss() + self.fc_final.loss()
        loss_nc = self.nc1.loss()
        loss_ncfc = self.ncfc1.loss()        
        #do loss ncfc


        for fc in self.fc2:
            loss_fc = loss_fc + fc.loss()
        for nc in self.nc2:
            loss_nc = loss_nc + nc.loss()
        for ncfc in self.ncfc2:
             loss_ncfc = loss_ncfc + ncfc.loss()

        return loss_fc, loss_nc, loss_ncfc



    def get_weights_nc(self):
        #weights_fc2 = [self.fc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.fc2))]
        weights_nc2 = [self.nc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.nc2))]
        avweights = [self.nc1.get_weight().detach().cpu().numpy()] + weights_nc2
        
        return avweights
    
    def get_weights_fc(self):
        weights_fc2 = [self.fc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.fc2))]
        avweights = [self.fc1.get_weight().detach().cpu().numpy()]+ weights_fc2 + [self.fc_final.get_weight().detach().cpu().numpy()]
        return avweights
    

    def get_weights_ncfc(self):
        #weights_fc2 = [self.fc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.fc2))]
        weights_ncfc2 = [self.ncfc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.ncfc2))]
        avweights = [self.ncfc1.get_weight().detach().cpu().numpy()] + weights_ncfc2
        
        return avweights




    def clamp_weight(self):
        #clamping weights for the convex part only
        self.fc_final.clamp_weights()
        #self.fc1.clamp_weights()
        for fc in self.fc2:
            fc.clamp_weights()
        #self.nc1.clamp_weights()
        #for nc in self.nc2:
        #    nc.clamp_weights()








if __name__ == '__main__':
   input_dim = 2
   output_dim = 1
   hidden_number_layers_details = [5,5]
   input_dim1 = 1
   hidden_number_layers_details1 = [2,2]
   model = PICNN_network(input_dim, output_dim, hidden_number_layers_details, input_dim1, hidden_number_layers_details1).to(device)
   x = torch.randn(10,2).to(device)
   y = torch.randn(10,1).to(device)
   z = model(x,y,epoch=1).to(device)
   print(z)
   loss = model.getLossRegL0()
   print(loss)

   print(model)
   print(model.fc1.get_weight())
    
   print(model.nc1.get_weight())


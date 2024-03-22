import torch
import numpy as np
    
class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,30)
        self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x2 = torch.tanh(self.L3(x2)) 

        x3 = torch.tanh(self.L4(x2))
        x3 = self.L5(x2)

        return x3
    
class NN_small(torch.nn.Module):
    def __init__(self):
        super(NN_small,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x3 = self.L5(x2)

        return x3

class NN_large(torch.nn.Module):
    def __init__(self):
        super(NN_large,self).__init__()
        self.L1 = torch.nn.Linear(2,100)
        self.L2 = torch.nn.Linear(100,100) 
        self.L3 = torch.nn.Linear(100,100)
        self.L4 = torch.nn.Linear(100,100)
        self.L5 = torch.nn.Linear(100,100)
        self.L6 = torch.nn.Linear(100,100)
        self.L7 = torch.nn.Linear(100,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x2 = torch.tanh(self.L3(x2))
        x3 = torch.tanh(self.L4(x2))
        x4 = torch.tanh(self.L5(x3))
        x5 = torch.tanh(self.L6(x4))

        x6 = self.L7(x5)

        return x6
    

def projection_softmax(net_values,low,high,a,hard_constraint=False):
    m = (high+low)/2.
    delta = high - m
    if not hard_constraint:
        b = 2*delta*(1+np.exp(-a*delta))/(1-np.exp(-a*delta))
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m
    else:
        b = 2*(high-m)
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m

    return out

def projection_clamp(net_values,low,high):
    out = torch.clamp(net_values,low,high)

    return out


def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    
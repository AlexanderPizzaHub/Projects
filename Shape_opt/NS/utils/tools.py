import torch
from torch.autograd import Variable
import numpy as np

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging


def from_numpy_to_tensor_with_grad(numpys,dtype=torch.float64):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=True).type(dtype)
        )

    return outputs

def from_numpy_to_tensor(numpys,dtype=torch.float64):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=False).type(dtype)
        )

    return outputs

def splitter(list_of_sets):
    tensorlist = [] 
    for item in list_of_sets:
        x1,x2 = item.T 
        tx1,tx2 = from_numpy_to_tensor_with_grad([x1.reshape(-1,1),x2.reshape(-1,1)])
        tensorlist.append([tx1,tx2])
    return tensorlist
    

class BFGS_hist(object):
    #Assert every stored data are torch.tensor.
    #first update, then precondition.
    def __init__(self,memory):
        #memory are heaps.
        self.x = []
        self.g = []
        self.s = []
        self.y = []
        self.rho = []
        self.memory = memory

    def update(self,x_new,g_new):
        if len(self.s)>=self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

        self.x.append(x_new)
        self.g.append(g_new)
        
        if len(self.x)>1:
            x_diff = self.x[-1] - self.x[-2]
            g_diff = self.g[-1] - self.g[-2]

            curv_cond = torch.mean(x_diff * g_diff)

            self.s.append(x_diff)
            self.y.append(g_diff)
            self.rho.append(1./torch.mean(g_diff * x_diff)) #assert L2 inner product
            print("BFGS history updated. The curvature condition is: {}",format(curv_cond))
            return curv_cond.detach().numpy()

    def precondition(self,grad):
        q = grad
        max_ind = len(self.s)
        if max_ind == 0:
            return -grad
        
        alpha = torch.zeros([max_ind,1])
        for ind in np.arange(0,max_ind):
            alpha[-ind-1] = self.rho[-ind-1] * torch.mean(self.s[-ind-1] * q)
            q = q - alpha[-ind-1] * self.y[-ind-1]
        gamma  = torch.mean(self.s[-1] * self.y[-1])/torch.mean(self.y[-1] * self.y[-1])
        z = gamma * q 

        beta = torch.zeros([max_ind,1])
        for ind in np.arange(max_ind-1,-1,-1):
            beta[-ind-1] = self.rho[-ind-1] * torch.mean(self.y[-ind-1] * z)
            z = z + self.s[-ind-1] * (alpha[-ind-1] - beta[-ind-1])

        z = -z 
        print("gradient preconditioned.")
        return z
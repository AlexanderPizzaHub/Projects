import torch
from torch.autograd import Variable
import numpy as np

#This version allows training multiple neural networks withing PDE.
#gpu version still debugging

def from_numpy_to_tensor(numpys,require_grads,dtype=torch.float64):
    #numpys: a list of numpy arrays.
    #requires_grads: a list of boolean to indicate whether give gradients
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]),requires_grad=require_grads[ind]).type(dtype)
        )

    return outputs
    

class BFGS_hist_old(object):
    #Assert every stored data are torch.tensor.
    #first update, then precondition.

    #did not give inner product on product Hilbert space
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

            curv_cond = torch.mean(g_diff * x_diff,dim=0,keepdim=True)

            self.s.append(x_diff)
            self.y.append(g_diff)
            self.rho.append(torch.divide(1.,torch.mean(g_diff * x_diff,dim=0,keepdim=True))) #assert L2 inner product
            npcond = curv_cond.detach().numpy()
            print("BFGS history updated. The curvature condition is: {}".format(npcond))
            return npcond

    def precondition(self,grad):
        q = grad
        max_ind = len(self.s)
        if max_ind == 0:
            return -grad
        
        alpha = torch.zeros([max_ind,1])
        for ind in np.arange(0,max_ind):
            #print(self.rho[-ind-1].shape,torch.mean(self.s[-ind-1] * q,dim=0).shape)
            alpha[ind] = torch.matmul(self.rho[-ind-1], torch.mean(self.s[-ind-1] * q,dim=0))
            q = q - alpha[ind] * self.y[-ind-1]
        gamma  = torch.mean(self.s[-1] * self.y[-1])/torch.mean(self.y[-1] * self.y[-1])
        z = gamma * q 

        beta = torch.zeros([max_ind,1])
        for ind in np.arange(max_ind-1,-1,-1):
            beta[-ind-1] = torch.matmul(self.rho[-ind-1], torch.mean(self.y[-ind-1] * z,dim=0))
            z = z + self.s[-ind-1] * (alpha[-ind-1] - beta[-ind-1])

        z = -z 
        print("gradient preconditioned.")
        return z
    
def product_Hilbert_inner(x1,x2):
    #assert shape (N,2)
    h1 = torch.mean(x1[:,0] * x2[:,0],dim=0,keepdim=True)
    h2 = torch.mean(x1[:,1] * x2[:,1],dim=0,keepdim=True)
    return h1+h2

class BFGS_hist(object):
    #Assert every stored data are torch.tensor.
    #first update, then precondition.

    #did not give inner product on product Hilbert space
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

            curv_cond1 = torch.mean(g_diff[:,0] * x_diff[:,0],dim=0,keepdim=True)
            curv_cond2 = torch.mean(g_diff[:,1] * x_diff[:,1],dim=0,keepdim=True)
            #curv_cond = torch.mean(g_diff * x_diff,dim=0,keepdim=True)
            curv_cond = curv_cond1 + curv_cond2

            self.s.append(x_diff)
            self.y.append(g_diff)
            #self.rho.append(torch.divide(1.,torch.mean(g_diff * x_diff,dim=0,keepdim=True))) 
            self.rho.append(1./curv_cond)
            #assert L2 inner product
            npcond = curv_cond.detach().numpy()
            print("BFGS history updated. The curvature condition is: {}".format(npcond))
            return npcond
        

    def precondition(self,grad):
        q = grad
        max_ind = len(self.s)
        if max_ind == 0:
            return -grad
        
        alpha = torch.zeros([max_ind,1])
        for ind in np.arange(0,max_ind):
            #print(self.rho[-ind-1].shape,torch.mean(self.s[-ind-1] * q,dim=0).shape)
            #alpha[ind] = torch.matmul(self.rho[-ind-1], torch.mean(self.s[-ind-1] * q,dim=0))
            alpha[-ind-1] = self.rho[-ind-1] * product_Hilbert_inner(self.s[-ind-1],q)
            q = q - alpha[-ind-1] * self.y[-ind-1]
        #gamma  = torch.mean(self.s[-1] * self.y[-1])/torch.mean(self.y[-1] * self.y[-1])
        gamma  = product_Hilbert_inner(self.s[-1],self.y[-1])/product_Hilbert_inner(self.y[-1],self.y[-1])
        z = gamma * q 

        beta = torch.zeros([max_ind,1])
        for ind in np.arange(0,max_ind):
            beta[ind] = self.rho[ind] * product_Hilbert_inner(self.y[ind], z)
            z = z + self.s[ind] * (alpha[ind] - beta[ind])

        z = -z 
        print("gradient preconditioned.")
        return z
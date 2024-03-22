import torch
import numpy as np
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from utils import model,tools,pde,validation
import os
import pickle as pkl
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
import yaml
from operator import itemgetter
from matplotlib.path import Path

torch.set_default_dtype(torch.float64)

with open("dataset/"+'4000points','rb') as pfile:
    d_c = pkl.load(pfile)
    b_out = pkl.load(pfile)
    b_int = pkl.load(pfile)

#b_out: [gamma_d1,gamma_c1,gamma_d2,gamma_c2,gamma_n,gamma_c3]. 
#b_int: [cir1,cir2,cir3]

dx1,dx2 = d_c.T 
print(len(dx1),len(dx2))
tdx = tools.from_numpy_to_tensor_with_grad([dx1.reshape(-1,1),dx2.reshape(-1,1)])
tb_o_list = tools.splitter(b_out)
tb_i_list = tools.splitter(b_int)

tbo_unsplitted = tools.from_numpy_to_tensor(b_out)
tbi_unsplitted = tools.from_numpy_to_tensor(b_int)

print(type(tb_i_list))

normal_o = [pde.compute_normal_nonclosed(bd_col) for bd_col in tbo_unsplitted]
normal_i = [pde.compute_normal(bd_col) for bd_col in tbi_unsplitted]


y = model.NN()

def f(x):
    return torch.zeros_like(x) #R2

def g(x):
    out = torch.zeros_like(x)
    out[:,1] += -10.0
    return out

E = 25 
nu = 0.45

mu = E/(2*(1+nu))
ld = E*nu/((1+nu)*(1-2*nu))

mse_loss = torch.nn.MSELoss()

#optimizer = torch.optim.LBFGS(y.parameters(),line_search_fn='strong_wolfe',max_iter=10000,max_eval=10000,tolerance_grad=1e-12,tolerance_change=1e-12,history_size=100) 

rec = validation.record_AONN()

def hook(optimizer,nploss):
    rec.updateEpoch()
    if rec.epoch % 100 == 0:
        print("epoch: ",rec.epoch,"loss",nploss)
        with torch.no_grad():
            y_val = y(tdx[0],tdx[1])
            y_val = torch.sqrt(y_val[:,0]**2 + y_val[:,1]**2)
            plt.figure(figsize=[16,10])
            plt.scatter(tdx[0].detach().numpy(),tdx[1].detach().numpy(),s=20,c=y_val.detach().numpy())
            plt.colorbar()
            plt.savefig('yscale3.png',dpi=200)
            plt.close()

#optimizer = torch.optim.Adam(y.parameters(),lr=0.0001)
optimizer = torch.optim.LBFGS(y.parameters(),line_search_fn='strong_wolfe',max_iter=30000,max_eval=30000,tolerance_grad=1e-12,tolerance_change=1e-12,history_size=100,stephook=hook)

bwlist = [1,1,1,1,1,1]
def closure():
    optimizer.zero_grad()

    pde_lhs = pde.pde(y,mu,ld,tdx[0],tdx[1])
    pde_rhs = f(pde_lhs)

    pres = mse_loss(pde_lhs,pde_rhs)

    bdry_lhs_all = pde.bdry(y,mu,ld,tb_o_list,tb_i_list,normal_o,normal_i)

    bdry_lhs = bdry_lhs_all[0]
    
    bdry_rhs = []


    # counter clockwise
    bdry_rhs.append(torch.zeros_like(bdry_lhs[0]))
    bdry_rhs.append(torch.zeros_like(bdry_lhs[1]))
    bdry_rhs.append(torch.zeros_like(bdry_lhs[2]))
    bdry_rhs.append(torch.zeros_like(bdry_lhs[3]))
    bdry_rhs.append(g(bdry_lhs[4]))
    bdry_rhs.append(torch.zeros_like(bdry_lhs[5]))


    bdry_lhs_in = bdry_lhs_all[1]
    bdry_rhs_in = []
    bdry_rhs_in.append(torch.zeros_like(bdry_lhs_in[0]))
    bdry_rhs_in.append(torch.zeros_like(bdry_lhs_in[1]))
    bdry_rhs_in.append(torch.zeros_like(bdry_lhs_in[2]))

    bres = 0.0 
    for i in [0,1,2,3,4,5]:
        bres += bwlist[i] * mse_loss(bdry_lhs[i],bdry_rhs[i])

    for i in [0,1,2]:
        bres += mse_loss(bdry_lhs_in[i],bdry_rhs_in[i])

    loss = pres + (bres)
    loss.backward()
    return loss.detach().numpy()



for epoch in range(1):
    optimizer.step(closure)
    

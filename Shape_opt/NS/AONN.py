import torch
import numpy as np
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from utils import model,tools,validation,pde
import os
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
import yaml
from operator import itemgetter

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['NS']['AONN']
    except yaml.YAMLError as exc:
        print(exc)

torch.set_default_dtype(torch.float64)
mse_loss = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datapath = 'dataset/'
dataname = config['dataname']

exppath = "results/AONN/" + config['expname'] + "/"


'''
All tunable params placed here
'''


bw,max_iter_out,max_iter_inner,eta,gamma = itemgetter('bw','out_iter','inner_iter','stepsize','decay_rate')(config)


max_iter_y ,max_iter_p,max_iter_u = max_iter_inner

val_interval = 200000
tol = 1e-16

nu = 1/50

y = model.NN()
p = model.pres()

ld = model.NN()
q = model.pres()

phi = model.deform()

y.apply(model.init_weights)
p.apply(model.init_weights)
ld.apply(model.init_weights)
q.apply(model.init_weights)
phi.apply(model.init_weights)

if not os.path.exists(exppath):
    os.makedirs(exppath)

if not os.path.exists(exppath+"u_plots/"):
    os.makedirs(exppath+"y_plots/")

if not os.path.exists(exppath+'u_plots/'):
    os.makedirs(exppath+"u_plots/")

if not os.path.exists(exppath+'p_plots/'):
    os.makedirs(exppath+"p_plots/")

with open("dataset/"+dataname,'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)


dx1,dx2 = d_c.T 
print(len(dx1),len(dx2))
tdx = tools.from_numpy_to_tensor_with_grad([dx1.reshape(-1,1),dx2.reshape(-1,1)])
tb_o_list = tools.splitter(b_c)

tbo_unsplitted = tools.from_numpy_to_tensor(b_c)




def f(x):
    return torch.zeros_like(x) #R2

def g(x):
    out = torch.zeros_like(x)
    u_in = 2.5*(1+x[:,1])*(1-x[:,1])
    out[:,0] = u_in
    return out



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
            plt.savefig('yscale2.png',dpi=200)
            plt.close()

#construct closure
def clear_gradients(optimizers):
    for item in optimizers:
        item.zero_grad()




info = validation.expInfo(config)
info.timelist = []
start = time.time()

#validation.plot_2D_scatter(y,tdx1.detach(),tdx2.detach(),exppath+'initial')
validation.plot_2D_scatter(tdx[0].detach(),tdx[1].detach(),exppath+'initial')

params_prim = list(y.parameters()) + list(p.parameters())
params_adj = list(ld.parameters()) + list(q.parameters())
bwlist = [100,100,100,100,100,100]

for epoch in range(max_iter_out):
    optimizer_prim = opt.LBFGS(params_prim,line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_y,tolerance_grad=tol,tolerance_change=tol)
    optimizer_adj = opt.LBFGS(params_adj,line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_p,tolerance_grad=tol,tolerance_change=tol)
    optimizer_phi = opt.LBFGS(phi.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_u,tolerance_grad=tol,tolerance_change=tol)
    
    normal_o = [pde.compute_normal_nonclosed(bd_col) for bd_col in tbo_unsplitted]

#For u, given c, it solves the primal pde
    
    print("solving state...")
    def closure_prim():
        clear_gradients([optimizer_prim,optimizer_adj,optimizer_phi])
        
        
        pde_lhs,div_u = pde.pde(y,p,nu,tdx[0],tdx[1])
        pde_rhs = f(pde_lhs)

        pres = mse_loss(pde_lhs,pde_rhs)

        bdry_lhs = pde.bdry(y,p,nu,tb_o_list,normal_o)
        
        #print(bdry_lhs[4][:,0])
        bdry_rhs = []

        # counter clockwise
        bdry_rhs.append(g(torch.concatenate([tb_o_list[0][0],tb_o_list[0][1]],dim=-1))) #gamma i
        bdry_rhs.append(torch.zeros_like(bdry_lhs[1])) #gamma w1
        bdry_rhs.append(torch.zeros_like(bdry_lhs[2])) #gamma o
        bdry_rhs.append(torch.zeros_like(bdry_lhs[3])) #gamma w2
        bdry_rhs.append(torch.zeros_like(bdry_lhs[4])) #gamma f
        bdry_rhs.append(torch.zeros_like(bdry_lhs[5])) #gamma w3

        bres = 0.0 
        for i in [0,1,2,3,4,5]:
            bres += bwlist[i] * mse_loss(bdry_lhs[i],bdry_rhs[i])


        dres = torch.mean(torch.square(div_u))
        loss = pres + bres + 10 * dres
        loss.backward()
        return loss.detach().numpy()
    
    optimizer_prim.step(closure_prim)



    print('solving adjoint...')
    with torch.no_grad():
        adj_rhs = y(tdx[0],tdx[1])-g(torch.concatenate([tdx[0],tdx[1]],dim=-1)) #In this example yd = gamma_i boundary.
    def closure_p():

        clear_gradients([optimizer_prim,optimizer_adj,optimizer_phi])
        
        pde_lhs,div_ld = pde.adjoint(ld,q,y,nu,tdx[0],tdx[1])

        pres = mse_loss(pde_lhs,adj_rhs)

        bdry_lhs = pde.bdry_adjoint(ld,q,y,nu,tb_o_list,normal_o)
        
        #print(bdry_lhs[4][:,0])
        bdry_rhs = []

        # counter clockwise
        bdry_rhs.append(torch.zeros_like(bdry_lhs[0])) #gamma i
        bdry_rhs.append(torch.zeros_like(bdry_lhs[1])) #gamma w1
        bdry_rhs.append(torch.zeros_like(bdry_lhs[2])) #gamma o
        bdry_rhs.append(torch.zeros_like(bdry_lhs[3])) #gamma w2
        bdry_rhs.append(torch.zeros_like(bdry_lhs[4])) #gamma f
        bdry_rhs.append(torch.zeros_like(bdry_lhs[5])) #gamma w3

        bres = 0.0 
        for i in [0,1,2,3,4,5]:
            bres += bwlist[i] * mse_loss(bdry_lhs[i],bdry_rhs[i])


        dres = torch.mean(torch.square(div_ld))
        loss = pres + bres + 10 * dres
        loss.backward()
        return loss.detach().numpy()

    optimizer_adj.step(closure_p)


    '''
    
    
    
    Regularization part has issue. Without regularization, correct optimal shape still can be obtained.



    
    '''

    '''pdedata_phi = torch.zeros([len(tdx[0]),2])
    print("proceeding to shape update...")
    def closure_phi():
        clear_gradients([optimizer_prim,optimizer_adj,optimizer_phi])

        pde_lhs = pde.reg(tdx[0],tdx[1],phi)

        pres = mse_loss(pde_lhs,pdedata_phi)


        gradient = pde.compute_gradient(y,ld,g(torch.concatenate([tb_o_list[4][0],tb_o_list[4][1]],dim=-1)),nu,tb_o_list[4][0],tb_o_list[4][1],normal_o[4])
        bdry_lhs = pde.bdry_misfit(ld,gradient,tb_o_list,normal_o)
        #print(bdry_lhs[4][:,0])
        bdry_rhs = []

        # counter clockwise
        bdry_rhs.append(torch.zeros_like(bdry_lhs[0])) #gamma i
        bdry_rhs.append(torch.zeros_like(bdry_lhs[1])) #gamma w1
        bdry_rhs.append(torch.zeros_like(bdry_lhs[2])) #gamma o
        bdry_rhs.append(torch.zeros_like(bdry_lhs[3])) #gamma w2
        bdry_rhs.append(torch.zeros_like(bdry_lhs[4])) #gamma f
        bdry_rhs.append(torch.zeros_like(bdry_lhs[5])) #gamma w3

        bres = 0.0 
        for i in [0,1,2,3,4,5]:
            bres += bwlist[i] * mse_loss(bdry_lhs[i],bdry_rhs[i])


        loss = pres + 10 * bres
        loss.backward()
        return loss.detach().numpy()

    optimizer_phi.step(closure_phi)'''
    
    
    '''update_domain = phi(tdx[0],tdx[1])
    tdx[0] = torch.tensor(tdx[0] + eta*update_domain[:,0].unsqueeze(-1),requires_grad=True)
    tdx[1] = torch.tensor(tdx[1] + eta*update_domain[:,1].unsqueeze(-1),requires_grad=True)'''

    for ind in [4]: #only gamma f is updatable
        #update_bdry = phi(tb_o_list[ind][0],tb_o_list[ind][1])
        gradient = pde.compute_gradient(y,ld,g(torch.concatenate([tb_o_list[4][0],tb_o_list[4][1]],dim=-1)),nu,tb_o_list[4][0],tb_o_list[4][1],normal_o[4])
        update_bdry = -torch.matmul(normal_o[4],gradient.unsqueeze(-1)).squeeze(-1)
        print(update_bdry.shape)

        tb_o_list[ind][0] = torch.tensor(tb_o_list[ind][0] + eta*update_bdry[:,0].unsqueeze(-1),requires_grad=True)
        tb_o_list[ind][1] = torch.tensor(tb_o_list[ind][1] + eta*update_bdry[:,1].unsqueeze(-1),requires_grad=True)


    with torch.no_grad():

        end = time.time()
        info.walltime = end-start
        info.endouteriter = epoch

        validation.plot_2D_with_boundary(tb_o_list,tdx[0],tdx[1],exppath+'iter{}'.format(epoch))
 
        info.timelist.append(end-start)
        info.saveinfo(exppath+"expInfo.json")
        print("On iteration {}, stepsize: {}".format(epoch,eta)) 
    

end = time.time() 

info.bestCost = float(np.min(rec.costhist))
info.bestValidation_y = float(np.min(rec.vhist_y))
info.bestValidation_u = float(np.min(rec.vhist_u))

#info.bestTraining = float(np.min(rec.losslist))
info.walltime = end-start
info.saveinfo(exppath+"expInfo.json")





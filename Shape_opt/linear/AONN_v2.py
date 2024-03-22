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


'''
Add a new step: approximate a mapping from initial domain to current domain
To tackle three issues:
1. enable a better sampling strategy because all intermediate steps are "stored" in this single mapping.
2. enable a better way to compute the normal vector field (with parameterized curve we can use tangential vector field to compute normal vector field).
3. (need to check) to better enforce constraints, like fixed boundary.

'''

with open("../config.yml", "r") as stream:
    try:
        ymlfile = yaml.safe_load(stream)
        config = ymlfile['linear']['AONN_v2']
    except yaml.YAMLError as exc:
        print(exc)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datapath = 'dataset/'
dataname = config['dataname']

exppath = "results/AONN/" + config['expname'] + "/"


'''
All tunable params placed here
'''


bw,max_iter_out,max_iter_inner,ld,gamma = itemgetter('bw','out_iter','inner_iter','stepsize','decay_rate')(config)


max_iter_y ,max_iter_p,max_iter_u = max_iter_inner

val_interval = 200000
tol = 1e-16
y = model.NN()
p = model.NN()
phi = model.deform()
deform = model.deform()

y.apply(model.init_weights)
p.apply(model.init_weights)
phi.apply(model.init_weights)
deform.apply(model.init_weights)



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
    c_c = pkl.load(pfile)


dx1,dx2 = np.split(d_c,2,axis=1)
bx1,bx2 = np.split(b_c,2,axis=1)
cx1,cx2 = np.split(c_c,2,axis=1)

tbx,tdx1,tdx2,tbx1,tbx2,tcx1,tcx2 = tools.from_numpy_to_tensor([b_c,dx1,dx2,bx1,bx2,cx1,cx2],[False,False,False,False,False,False,False])

def f_gen(x1,x2):
    return 2.5*(x1+0.4-x2**2)**2 +x1**2 + x2**2 -1

def p_gen(x1,x2):
    return torch.ones_like(x1,requires_grad=True)

bdrydat_prim = torch.zeros([len(bx1),1])
bdrydat_adj = torch.zeros([len(bx1),1])

rec = validation.record_AONN()

'''def hook_y(optimizer,nploss):
    stateitems = list(optimizer.state.items())
    epoch = stateitems[0][1]['n_iter']
    if epoch%val_interval==0:
        pdedata_y = f + u(tdx1,tdx2)
        with torch.enable_grad():
            loss,res,misfit = pde.pdeloss(y,tdx1,tdx2,pdedata_y,tbx1,tbx2,bdrydat_prim,bw)
        rec.updatePL(nploss,res[0].detach().numpy(),res[1].detach().numpy())
        print("Running u optimization at epoch {}...".format(epoch))


def hook_p(optimizer,nploss):
    stateitems = list(optimizer.state.items())
    epoch = stateitems[0][1]['n_iter']
    if epoch%val_interval ==0:
        pdedata_phi = y(tdx1,tdx2) - y_dat
        with torch.enable_grad():
            loss,apde,abc = pde.adjloss(p,tdx1,tdx2,pdedata_phi,tbx1,tbx2,bdrydat_adj,bw)
        rec.updateAL(nploss,apde.detach().numpy(),abc.detach().numpy())
        print("Running p optimization at epoch {}...".format(epoch))'''

def hook(optimizer,nploss):
    return

#construct closure
def clear_gradients(optimizers):
    for item in optimizers:
        item.zero_grad()

mse_loss = torch.nn.MSELoss()


info = validation.expInfo(config)
info.timelist = []


start = time.time()
normal = pde.compute_normal(tbx)


#validation.plot_2D_scatter(y,tdx1.detach(),tdx2.detach(),exppath+'initial')
validation.plot_2D_scatter_with_vec(tdx1.detach(),tdx2.detach(),tbx1.detach(),tbx2.detach(),normal,exppath+'initial')

for epoch in range(max_iter_out):
    optimizer_y = opt.LBFGS(y.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_y,tolerance_grad=tol,tolerance_change=tol)
    optimizer_p = opt.LBFGS(p.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_p,tolerance_grad=tol,tolerance_change=tol)
    optimizer_phi = opt.LBFGS(phi.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_u,tolerance_grad=tol,tolerance_change=tol)
    optimizer_df = opt.LBFGS(deform.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_u,tolerance_grad=tol,tolerance_change=tol)


    with torch.no_grad():
        update_d = deform(tdx1,tdx2)
        update_b = deform(tbx1,tbx2)
    tdx1_df = torch.tensor(tdx1+update_d[:,0].unsqueeze(-1),requires_grad=True)
    tdx2_df = torch.tensor(tdx2+update_d[:,1].unsqueeze(-1),requires_grad=True)
    tbx1_df = torch.tensor(tbx1+update_b[:,0].unsqueeze(-1),requires_grad=True)
    tbx2_df = torch.tensor(tbx2+update_b[:,1].unsqueeze(-1),requires_grad=True)

#For u, given c, it solves the primal pde
    with torch.no_grad():
        pdedata_y = f_gen(tdx1_df,tdx2_df)
        
    def closure_y():

        clear_gradients([optimizer_y,optimizer_p,optimizer_phi])
        
        
        #loss= pde.pdeloss(u,tdx1,tdx2,,bc_x_0,bc_t,u(bc_x_L,bc_t),ic_x,ic_t,init_u,pw,bw,iw)
        loss,res,misfit = pde.pdeloss(y,tdx1_df,tdx2_df,pdedata_y,tbx1_df,tbx2_df,bdrydat_prim,bw)
        loss.backward()
        
        return loss
    
    for e1 in range(1):
        optimizer_y.step(closure_y)
        #losslist_u.append(loss)

    with torch.no_grad():
        pdedata_adj = p_gen(tdx1_df,tdx2_df)

    def closure_p():

        clear_gradients([optimizer_y,optimizer_p,optimizer_phi])
        
        loss,apde,abc = pde.adjloss(p,tdx1_df,tdx2_df,pdedata_adj,tbx1_df,tbx2_df,bdrydat_adj,bw)
        loss.backward()
        
        return loss

    for e2 in range(1):
        optimizer_p.step(closure_p)
        #losslist_phi.append(loss)

    pdedata_phi = torch.zeros([len(tdx1_df),2])

    normal = pde.compute_normal(torch.concatenate([tbx1_df,tbx2_df],dim=1)).detach()
    print(normal.shape)

    print("proceeding to shape update...")
    def closure_phi():
        clear_gradients([optimizer_y,optimizer_p,optimizer_phi])

        loss,ppde,pbc = pde.regloss(phi,y,p,tdx1_df,tdx2_df,pdedata_phi,tbx1_df,tbx2_df,bw,normal)

        loss.backward()
        return loss

    optimizer_phi.step(closure_phi)
    
    with torch.no_grad():
        update_bdry = phi(tbx1_df,tbx2_df)
        update_domain = phi(tdx1_df,tdx2_df)

        dxdat = deform(tdx1,tdx2) - ld*(-update_domain)
        bxdat = deform(tbx1,tbx2) - ld*(-update_bdry)

        update_deform = torch.concatenate([dxdat,bxdat],dim=0)

        '''dx1dat = tdx1 + ld*update_domain[:,0].unsqueeze(-1)
        dx2dat = tdx2 + ld*update_domain[:,1].unsqueeze(-1)
        bx1dat = tbx1 + ld*update_bdry[:,0].unsqueeze(-1)
        bx2dat = tbx2 + ld*update_bdry[:,1].unsqueeze(-1)'''

    def closure_deform():
        clear_gradients([optimizer_df])

        deform_out_d = deform(tdx1,tdx2)
        deform_out_b = deform(tbx1,tbx2)
        deform_out = torch.concatenate([deform_out_d,deform_out_b],dim=0)
        loss = mse_loss(deform_out,update_deform)
        loss.backward()
        return loss
        
    optimizer_df.step(closure_deform)


    with torch.no_grad():
        #rec.validate(y,u)
        #info.lastvy = float(rec.vhist_y[-1])
        #info.lastvu = float(rec.vhist_u[-1])
        #info.vylist.append(info.lastvy)
        #info.vulist.append(info.lastvu)
        #validation.plot_2D_scatter(y,tdx1.detach(),tdx2.detach(),exppath+'iter{}phi'.format(epoch))
        end = time.time()
        info.walltime = end-start
        info.endouteriter = epoch
        normal = pde.compute_normal(torch.concatenate([tbx1_df,tbx2_df],dim=1)).detach()
        validation.plot_2D_scatter_with_vec(tdx1_df.detach(),tdx2_df.detach(),tbx1_df.detach(),tbx2_df.detach(),normal,exppath+'iter{}phi'.format(epoch))
        #validation.plot_vecfield(tdx1.detach(),tdx2.detach(),normal,exppath+'iter{}phi'.format(epoch))
        info.timelist.append(end-start)
        info.saveinfo(exppath+"expInfo.json")
        print("On iteration {}, stepsize: {}".format(epoch,ld)) 
        print(tdx1_df.mean(),tdx2_df.mean())
    

end = time.time() 

info.bestCost = float(np.min(rec.costhist))
info.bestValidation_y = float(np.min(rec.vhist_y))
info.bestValidation_u = float(np.min(rec.vhist_u))

#info.bestTraining = float(np.min(rec.losslist))
info.walltime = end-start
info.saveinfo(exppath+"expInfo.json")





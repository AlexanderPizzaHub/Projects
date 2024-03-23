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
        config = ymlfile['linear']['AONN_BFGS']
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

alpha = 0.01

bw,max_iter_out,max_iter_inner,ld,gamma,m = itemgetter('bw','out_iter','inner_iter','stepsize','decay_rate','memory')(config)


max_iter_y ,max_iter_p,max_iter_u = max_iter_inner

val_interval = 200000
tol = 1e-16
y = model.NN()
p = model.NN()
u = model.NN()

y.apply(model.init_weights)
p.apply(model.init_weights)
u.apply(model.init_weights)

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

tdx1,tdx2,tbx1,tbx2,tcx1,tcx2 = tools.from_numpy_to_tensor([dx1,dx2,bx1,bx2,cx1,cx2],[True,True,False,False,False,False])



with open("dataset/gt_on_{}".format(dataname),'rb') as pfile:
    y_gt = pkl.load(pfile)
    u_gt = pkl.load(pfile)
    p_gt = pkl.load(pfile)
    y_dat_np = pkl.load(pfile)
    f_np = pkl.load(pfile)
    bdnp = pkl.load(pfile)

f,y_dat,bdrydat_prim = tools.from_numpy_to_tensor([f_np,y_dat_np,bdnp],[False,False,False])

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
    stateitems = list(optimizer.state.items())
    epoch = stateitems[0][1]['n_iter']
    if epoch%val_interval == 0:
        pdedata_phi = y(tdx1,tdx2) - y_dat
        pdedata_y = f + u(tdx1,tdx2)
        with torch.enable_grad():
            lossp,res,misfit = pde.pdeloss(y,tdx1,tdx2,pdedata_y,tbx1,tbx2,bdrydat_prim,bw)
            lossa,apde,abc = pde.adjloss(p,tdx1,tdx2,pdedata_phi,tbx1,tbx2,bdrydat_adj,bw)
            cost = pde.cost_mse(y,y_dat,u,alpha,tdx1,tdx2)
        rec.updatePL(lossp.detach().numpy(),res[0].detach().numpy(),res[1].detach().numpy())
        rec.updateAL(lossa.detach().numpy(),apde.detach().numpy(),abc.detach().numpy())
        rec.updateCL(cost.detach().numpy())

#construct closure
def clear_gradients(optimizers):
    for item in optimizers:
        item.zero_grad()

mse_loss = torch.nn.MSELoss()


info = validation.expInfo(config)
info.timelist = []
info.vylist = []
info.vulist = []
info.memory = m
info.curvcon = []
start = time.time()

bfgshist = tools.BFGS_hist(m)

for epoch in range(max_iter_out):
    optimizer_y = opt.LBFGS(y.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_y,tolerance_grad=tol,tolerance_change=tol)
    optimizer_p = opt.LBFGS(p.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_p,tolerance_grad=tol,tolerance_change=tol)
    optimizer_u = opt.LBFGS(u.parameters(),line_search_fn='strong_wolfe',stephook=hook,max_iter=max_iter_u,tolerance_grad=tol,tolerance_change=tol)
    
#For u, given c, it solves the primal pde
    def closure_y():

        clear_gradients([optimizer_y,optimizer_p,optimizer_u])
        
        with torch.no_grad():
            pdedata_y = f + u(tdx1,tdx2)
        #loss= pde.pdeloss(u,tdx1,tdx2,,bc_x_0,bc_t,u(bc_x_L,bc_t),ic_x,ic_t,init_u,pw,bw,iw)
        loss,res,misfit = pde.pdeloss(y,tdx1,tdx2,pdedata_y,tbx1,tbx2,bdrydat_prim,bw)
        loss.backward()
        
        return loss

    def closure_p():

        clear_gradients([optimizer_y,optimizer_p,optimizer_u])
        
        with torch.no_grad():
            pdedata_phi = y(tdx1,tdx2) - y_dat

        loss,apde,abc = pde.adjloss(p,tdx1,tdx2,pdedata_phi,tbx1,tbx2,bdrydat_adj,bw)
        loss.backward()
        
        return loss

    u_step = None

    def closure_ctrl():
        #Gradient step by least squares
        #In burgers, the gradient is given explicit by -\phi(x,0)
        #Here using dc_x and zeros t as collocations
        clear_gradients([optimizer_y,optimizer_p,optimizer_u])
        loss = mse_loss(u(tdx1,tdx2),u_step)
        loss.backward()
        return loss
    
    for e1 in range(1):
        optimizer_y.step(closure_y)
        #losslist_u.append(loss)
            
    for e2 in range(1):
        optimizer_p.step(closure_p)
        #losslist_phi.append(loss)


    #grad = p + alpha*u
    with torch.no_grad():
        u_dat = u(tdx1,tdx2)
        p_dat = p(tdx1,tdx2)
        grad_J = p_dat + alpha*u_dat
        cc = bfgshist.update(u_dat,grad_J)
        if cc is not None:
            info.curvcon.append(float(cc))
        u_step = u(tdx1,tdx2) + ld * bfgshist.precondition(grad_J)

    for e3 in range(1):
        if ld > 1e-4:
            ld = gamma*ld
        optimizer_u.step(closure_ctrl)
    
    end = time.time()
    info.walltime = end-start
    info.endouteriter = epoch
    with torch.no_grad():
        rec.validate(y,u)
        info.lastvy = float(rec.vhist_y[-1])
        info.lastvu = float(rec.vhist_u[-1])
        info.vylist.append(info.lastvy)
        info.vulist.append(info.lastvu)
        info.timelist.append(end-start)
        info.saveinfo(exppath+"expInfo.json")
        print("On iteration {}, stepsize: {}".format(epoch,ld)) 
    

end = time.time() 

info.bestCost = float(np.min(rec.costhist))
info.bestValidation_y = float(np.min(rec.vhist_y))
info.bestValidation_u = float(np.min(rec.vhist_u))

#info.bestTraining = float(np.min(rec.losslist))
info.walltime = end-start
info.saveinfo(exppath+"expInfo.json")





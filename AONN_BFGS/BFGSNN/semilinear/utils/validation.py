import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json
from . import groundtruth as gt
from . import tools

dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = 20
val_x1=np.arange(0,1,1/resolution).reshape(-1,1)
val_x2=np.arange(0,1,1/resolution).reshape(-1,1)
t_vx1 = Variable(torch.from_numpy(val_x1)).type(dtype).to(device)
t_vx2 = Variable(torch.from_numpy(val_x2)).type(dtype).to(device)

alpha = 0.01


#Generate grids to output graph
val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)
plot_val_x1 = np.ravel(val_ms_x1).reshape(-1,1)
plot_val_x2 = np.ravel(val_ms_x2).reshape(-1,1)

t_val_vx1,t_val_vx2 = tools.from_numpy_to_tensor([plot_val_x1,plot_val_x2],[False,False],dtype=dtype)

y_gt,u_gt,p_gt,y_data,f = gt.data_gen_interior(np.concatenate([plot_val_x1,plot_val_x2],axis=1))

t_ygt,t_ugt,t_pgt,t_yd = tools.from_numpy_to_tensor([y_gt,u_gt,p_gt,y_data],[False,False,False,False],dtype=dtype)



def plot_2D(net,path):
    data = net(t_val_vx1,t_val_vx2).detach().numpy().reshape([resolution,resolution])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(val_ms_x1,val_ms_x2,data, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()

def plot_2D_with_proj(net,projector,path,low,high):
    pt_u = projector(net(t_val_vx1,t_val_vx2)/(-alpha),low,high).detach().numpy().reshape([resolution,resolution])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(val_ms_x1,val_ms_x2,pt_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(path)
    plt.close()

mse_loss = torch.nn.MSELoss()
y_L2 = torch.sqrt(torch.mean(torch.square(t_ygt)))
u_L2 = torch.sqrt(torch.mean(torch.square(t_ugt)))


'''
In coupled method, it solves primal state and adjoint state at the same time. 
Each state contains a boundary loss and pde loss, which will be recorded.
Adding these two loss under given penalty weight, it gives the total loss, alse being recorded.
We should also evaluate the cost, although we do not use it during training.

For validation, the primal state groundtruth and contrl groundtruth will be used.
The validation error has two component: relative L2 and relative L_infinity, and validation grid is on fixed 50*50 grid.
'''
class record_couple(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()
        self.vhist_y = list()
        self.vhist_u = list()
        self.costhist = list()
        self.epoch = 0
        self.best_pinnloss = 999999.0

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,al,apde,abc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)
    
    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_y,vl_u):
        self.vhist_y.append(vl_y)
        self.vhist_u.append(vl_u)
    
    def validate(self,y,u):
        with torch.no_grad():
            vy = (torch.sqrt(mse_loss(y(t_val_vx1,t_val_vx2),t_ygt))/y_L2).detach().numpy()
            vu = (torch.sqrt(mse_loss(u(t_val_vx1,t_val_vx2)/(-alpha),t_ugt))/u_L2).detach().numpy()

        self.updateVL(vy,vu)
        return vy,vu

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.adjhist,self.vhist_y,self.vhist_u]
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])
        plt.subplot(231)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(232)
        plt.loglog(self.pdehist)
        plt.loglog(self.adjhist)
        plt.legend(['pde','adj'])
        plt.title("pde/adj loss")

        plt.subplot(233)
        plt.loglog(self.vhist_y)
        plt.loglog(self.vhist_u)
        plt.legend(['state validation','control validation'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
In penalty method, it involves no adjoint equations. So there wil be only one boundary/pde record.
But in addition, here will be cost objective.

The validation is same as above.
'''
class record_penalty(object):
    def __init__(self):
        self.losslist = list()
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()
        self.vhist_y = list()
        self.vhist_u = list()
        self.epoch = 0
        self.costhist = list()

    def updateTL(self,loss):
        self.epoch= self.epoch+1
        self.losslist.append(loss)
        

    def updatePL(self,pl,ppde,pbc,cost):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        self.costhist.append(cost)

    def updateVL(self,vl_y,vl_u):
        self.vhist_y.append(vl_y)
        self.vhist_u.append(vl_u)
    
    def validate(self,y,u):
        #In penalty, ctrl itself is NN and do not need projection.
        with torch.no_grad():
            print(y_gt.shape)
            vy = (torch.sqrt(mse_loss(y(t_val_vx1,t_val_vx2),t_ygt))/y_L2).detach().numpy()
            vu = (torch.sqrt(mse_loss(u(t_val_vx1,t_val_vx2),t_ugt))/u_L2).detach().numpy()

        self.updateVL(vy,vu)
        return vy,vu

    def getepoch(self):
        return self.epoch
    def getattr(self):
        return [self.losslist,self.pdehist,self.costhist,self.vhist_y,self.vhist_u]
    
    def plotinfo(self,path):
        plt.subplots(4,figsize=[20,20])
        plt.subplot(221)
        plt.loglog(self.losslist)
        plt.title("total loss")

        plt.subplot(222)
        plt.loglog(self.pdehist)
        plt.loglog(self.costhist)
        plt.legend(['pde loss','cost objective'])
        plt.title("pde loss/cost")

        plt.subplot(223)
        plt.loglog(self.vhist_y)
        plt.loglog(self.vhist_u)
        plt.title("validation")
        plt.legend(['state validation', 'control validation'])

        plt.subplot(224)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.legend(['pde residual','boundary residual'])
        plt.title("pinn loss")

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)


'''
In AONN method, primal state solve and adjoint solve are splitted. Each of them contains pde loss and boundary loss, 
which is be recorded seperately. The record is using individual epoch.

The epoch records the outer loop.

Similar to coupled method, the cost objective will not be used, but still be recorded.

Particularly, in AONN, the control NN is learned from the project GD, hence it is always admissible; there is no total loss.
'''
class record_AONN(object):
    def __init__(self):
        self.pdehist = list()
        self.pderes = list()
        self.pdebc = list()

        self.adjhist = list()
        self.adjres = list()
        self.adjbc = list()

        self.vhist_y = list()
        self.vhist_u = list()

        self.costhist = list()
        self.epoch = 0

    def updateEpoch(self):
        self.epoch= self.epoch+1
        

    def updatePL(self,pl,ppde,pbc):
        self.pdehist.append(pl)
        self.pderes.append(ppde)
        self.pdebc.append(pbc)
        
    def updateAL(self,al,apde,abc):
        self.adjhist.append(al)
        self.adjres.append(apde)
        self.adjbc.append(abc)

    def updateCL(self,cl):
        self.costhist.append(cl)

    def updateVL(self,vl_y,vl_u):
        self.vhist_y.append(vl_y)
        self.vhist_u.append(vl_u)
    
    def validate(self,y,u):
        with torch.no_grad():
            vy = np.sqrt(mse_loss(y(t_val_vx1,t_val_vx2),t_ygt).detach().numpy())/y_L2
            vu = np.sqrt(mse_loss(u(t_val_vx1,t_val_vx2),t_ugt).detach().numpy())/u_L2

        self.updateVL(vy,vu)
        return vy,vu
    
    def plotinfo(self,path):
        plt.subplots(6,figsize=[30,20])

        plt.subplot(231)
        plt.loglog(self.pdehist)
        plt.title("primal state loss")

        plt.subplot(232)
        plt.loglog(self.adjhist)
        plt.title('adjoint state loss')

        plt.subplot(233)
        plt.loglog(self.vhist_y)
        plt.loglog(self.vhist_u)
        plt.legend(['state validation','control validation'])
        plt.title("validation")

        plt.subplot(234)
        plt.loglog(self.costhist)
        plt.title("cost objective")

        plt.subplot(235)
        plt.loglog(self.pderes)
        plt.loglog(self.pdebc)
        plt.title("primal state loss")
        plt.legend(['pde residual','boundary condition'])

        plt.subplot(236)
        plt.loglog(self.adjres)
        plt.loglog(self.adjbc)
        plt.title('adjoint state loss')
        plt.legend(['pde residual','boundary condition'])

        plt.savefig(path+'history.png')
        plt.close()

        with open(path+"hist.pkl",'wb') as pfile:
            pkl.dump(self,pfile)



'''
This is a general info recorder for the experiment.
'''
class expInfo(object):
    def __init__(self,dict = None):
        self.__dict__.update(dict)

    def saveinfo(self,path):
        info = json.dumps(self.__dict__,indent=4,separators=(',',':'))
        f = open(path,'w')
        f.write(info)
        f.close()
        return True
    
import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pickle as pkl
import json
from mpl_toolkits import mplot3d
from . import groundtruth as gt
from . import tools

dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = 20
val_x1=np.arange(-3,3,(6)/resolution).reshape(-1,1)
val_x2=np.arange(-3,3,(6)/resolution).reshape(-1,1)
t_vx1 = Variable(torch.from_numpy(val_x1)).type(dtype).to(device)
t_vx2 = Variable(torch.from_numpy(val_x2)).type(dtype).to(device)

alpha = 0.01


#Generate grids to output graph
val_ms_x1, val_ms_x2 = np.meshgrid(val_x1, val_x2)
tmp_vx1 = np.ravel(val_ms_x1)
tmp_vx2 = np.ravel(val_ms_x2)

r = np.sqrt(np.power(tmp_vx1,2)+np.power(tmp_vx2,2))
domain_mark = np.logical_and(r>=1.0,r<=3.0)
#domain_mark = np.logical_not(np.logical_and(np.logical_and(tmp_vx1>=0.25,tmp_vx1<=0.75),np.logical_and(tmp_vx2>=0.25,tmp_vx2<=0.75)))
plot_val_x1 = tmp_vx1[domain_mark].reshape(-1,1)
plot_val_x2 = tmp_vx2[domain_mark].reshape(-1,1)

t_val_vx1,t_val_vx2 = tools.from_numpy_to_tensor([plot_val_x1,plot_val_x2],[False,False],dtype=dtype)

y_gt,u_gt,p_gt,y_data,f = gt.data_gen_interior(np.concatenate([plot_val_x1,plot_val_x2],axis=1))

t_ygt,t_ugt,t_pgt,t_yd = tools.from_numpy_to_tensor([y_gt,u_gt,p_gt,y_data],[False,False,False,False],dtype=dtype)
def plot_2D_scatter(net,path):
        values = net(t_val_vx1,t_val_vx2).detach().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(plot_val_x1, plot_val_x2, values, c=values, cmap='Greens')
        plt.savefig(path)
        plt.close()

def plot_2D_trisurf(net,path):
        values = net(t_val_vx1,t_val_vx2).detach().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(plot_val_x1, plot_val_x2, values, c=values, cmap='viridis')
        plt.savefig(path)
        plt.close()

def plot_2D_scatter_with_projection(net,projector,path,low,high):
        values = projector(net(t_val_vx1,t_val_vx2)/(-alpha),low,high).detach().numpy()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(plot_val_x1, plot_val_x2, values, c=values, cmap='Greens')
        plt.savefig(path)
        plt.close()

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
    
    
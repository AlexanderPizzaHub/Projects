import jax.numpy as jnp
from jax import grad, vmap
import jax 

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax.numpy.linalg import lstsq

class recorder(object):
    def __init__(self,Nv):
        self.losslist = [] 
        self.epoch = 0 

        self.Nv=Nv

        val_x=jnp.arange(0,1,1/Nv).reshape([Nv])
        val_y=jnp.arange(0,1,1/Nv).reshape([Nv])

        #Generate grids to output graph
        val_ms_x, val_ms_y = jnp.meshgrid(val_x, val_y)
        plot_val_x = jnp.ravel(val_ms_x).reshape(-1,1)
        plot_val_y = jnp.ravel(val_ms_y).reshape(-1,1)

        self.coor = jnp.hstack([plot_val_x,plot_val_y])
        

    def pre_push(self,model,params,h,x):
        ''' 
        value of gramian at given point x.
        h:[p,1]
        '''
        d_theta_u = grad(model)(params,x)
        d_theta_u_flattened, unravel = jax.flatten_util.ravel_pytree(d_theta_u)
        h_flattened, unravel = jax.flatten_util.ravel_pytree(h)


        #print(h_flattened.T.shape,d_theta_u_flattened.shape)
        output = jnp.matmul(h_flattened,d_theta_u_flattened)
        return output 
    
    def push(self,model,params,h,step):
        v_pre_push = vmap(self.pre_push,(None,None,None,0))
        push_points = v_pre_push(model,params,h,self.coor)
        return push_points * step

    def draw_data(self,data,path):
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        im = ax.imshow(data,cmap=cm.coolwarm)
        ax.set_title('residum')
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax,shrink=0.82,pad=0.03,spacing="proportional")
        plt.tight_layout(h_pad=0)
        plt.savefig(path,dpi=200)
        plt.close()


    def draw(self,u,params,ugt,h,step,res,path):
        ''' 
        Draw the triple (residual u_theta-u_star, pushforward of NGD, pushforward of GD or ADAM)
        '''
        u_val = vmap(u,(None,0))(params,self.coor)
        ugt_val = vmap(ugt)(self.coor)

        residum = u_val - ugt_val

        pf = self.push(u,params,h,step)

        fig,ax = plt.subplots(1,3,figsize=(15,5))

        plt.subplot(1,3,1)
        im = ax[0].imshow(jnp.reshape(residum,(self.Nv,self.Nv)),cmap=cm.coolwarm)
        ax[0].set_title('residum')
        ax[0].set_axis_off()
        cbar = fig.colorbar(im, ax=ax[0],shrink=0.82,pad=0.03,spacing="proportional")

        plt.subplot(1,3,2)
        im = ax[1].imshow(jnp.reshape(pf,(self.Nv,self.Nv)),cmap=cm.coolwarm)
        ax[1].set_title('pushforward_step{}'.format(step))
        ax[1].set_axis_off()
        cbar = fig.colorbar(im, ax=ax[1],shrink=0.82,pad=0.03,spacing="proportional")

        plt.subplot(1,3,3)
        plt.plot(res)
        plt.title('residual norm')
        

        plt.tight_layout(h_pad=0)
        plt.savefig(path,dpi=200)
        plt.close()
        return True
    

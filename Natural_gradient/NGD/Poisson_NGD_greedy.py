import jax.numpy as jnp
from jax import grad, jit, vmap,hessian
from jax import random
import jax 
import pickle as pkl
import optax
from matplotlib import cm
import matplotlib as mpl
import os 
import matplotlib.pyplot as plt
from jax.numpy.linalg import lstsq
import pandas as pd
from utils import tools,pde,model,validation
from time import time

with open("/Users/dual/Documents/projects/Natural_gradient/NGD/dataset/5000points",'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)
    c_c = pkl.load(pfile)

print(d_c.shape,b_c.shape)

path = './NGDpoisson_t1'

key = random.PRNGKey(0)

initializer = jax.nn.initializers.glorot_uniform()

layers = [2,30,1]
params = model.init_network_params(key,layers)
u = model.NN(jnp.tanh)

tau = 100
greedy_step = 20

f = lambda x: 2*jnp.pi**2*(jnp.sin(jnp.pi*x[0])*jnp.sin(jnp.pi*x[1]))
u_d = lambda x: (jnp.sin(jnp.pi*x[0])*jnp.sin(jnp.pi*x[1]))

lap_func = lambda params: pde.laplace(lambda x: u(params,x))

#v_lap = jit(vmap(lap_func,(None,0)))

residual_dom = lambda params,x: 0.5*(lap_func(params)(x) + f(x))**2
residual_bdry = lambda params,x: 0.5*(u(params,x)-0)**2 

v_residual_dom = jit(vmap(residual_dom,(None,0)))
v_residual_bdry = jit(vmap(residual_bdry,(None,0)))

@jit
def loss(params):
    return jnp.mean(v_residual_dom(params,d_c)) + tau * jnp.mean(v_residual_bdry(params,b_c))



rec = validation.recorder(64)


grid = jnp.linspace(0, 30, 31)
steps = (0.5**grid)*0.001
ls_update = tools.grid_line_search_factory(loss, steps)


val_res =jit(lambda params: vmap(lambda x: (u(params,x)-u_d(x))**2,(0))(d_c))


if not os.path.exists(path):
    os.makedirs(path)

losslist = [] 
#condlist = [] 
stepsizelist = []
start = time()
for iteration in jnp.arange(0,1000):
    grads = grad(loss)(params)
    g = tools.gram(u,params,d_c)
    
    updates,res = tools.preconditioner_greedy(g,grads,greedy_step,1)    #If greedy, use this line
    #updates = tools.preconditioner(g,grads,-1)    #If no greedy applied, use this line
    #params = optax.apply_updates(params, updates) #If no line search, use this line
    params, actual_step = ls_update(params,updates) #If line search, use this line
    
    if iteration % 10 == 0:
        # errors
        l2_error = jnp.sqrt(jnp.mean(val_res(params)))
        #rec.draw(u,params,f,updates,actual_step,path+f'/NGD_{iteration}.png')
        losslist.append(l2_error)
        #condlist.append(cond)
        stepsizelist.append(actual_step)
        now = time()
        print(
            f'NGD Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error} '
            f' with stepsize: {actual_step} '
            f' in time: {now-start} '
            #f'condition number: {cond} '
            #f'selected basis: {ind} ' 
        )
        '''corr = jnp.corrcoef(g_col)
        rec.draw_data(corr,path+f'/corr_{iteration}.png')

        abscov = jnp.abs(corr)
        rec.draw_data(abscov,path+f'/abscorr_{iteration}.png')

        cov = jnp.cov(g_col)
        rec.draw_data(cov,path+f'/cov_{iteration}.png')

        plt.plot(g_col_norm)
        plt.savefig(path+f'/norm_{iteration}.png')
        plt.close()'''
        

with open('losshist_ngd_t1.pkl','wb') as pfile:
    pkl.dump(losslist,pfile)
    #pkl.dump(condlist,pfile)
    pkl.dump(stepsizelist,pfile)


'''  
NGD will stuck at local minima once the tangent space becomes not informative. Tangent vectors becomes highly correlated, preconditioner becomes ill-conditioned, and projecting turth gradient onto tangent space leads to huge loss of information.(large projection error).
'''



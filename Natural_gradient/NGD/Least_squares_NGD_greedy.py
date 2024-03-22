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
import numpy as np
from jax import config
from utils import tools, validation,model

config.update("jax_enable_x64", True)
key = random.PRNGKey(0)


path = './convergence_iter_largenet/gs20_deeper/'
layers = [2,20,20,1]
max_iter = 500
val_int = 10
greedy_step = 20


with open("dataset/5000points",'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)
    c_c = pkl.load(pfile)


params = model.init_network_params(key,layers)
u = model.NN(jnp.tanh)



f = lambda x: (jnp.sin(jnp.pi*x[0])*jnp.sin(jnp.pi*x[1]))+jnp.sin(jnp.pi*3/2*x[0])*jnp.sin(jnp.pi*3/2*x[1])

residual = lambda params,x: 0.5*(u(params,x) - f(x))**2
v_residual = jit(vmap(residual,(None,0)))

@jit
def loss(params):
    return jnp.mean(v_residual(params,d_c))


grid = jnp.linspace(0, 30, 31)
steps = (0.5**grid)
ls_update = tools.grid_line_search_factory(loss, steps)

rec = validation.recorder(64)


if not os.path.exists(path):
    os.makedirs(path)

losslist = []
for iteration in jnp.arange(0,max_iter):
    grads = grad(loss)(params)
    g = tools.gram(u,params,d_c)
    
    updates,res = tools.preconditioner_greedy(g,grads,greedy_step,1)    #If greedy, use this line
    #updates = tools.preconditioner(g,grads,-1)    #If no greedy applied, use this line
    #params = optax.apply_updates(params, updates) #If no line search, use this line
    params, actual_step = ls_update(params,updates) #If line search, use this line
    
    if iteration % val_int == 0:
        # errors
        l2_error = jnp.sqrt(loss(params))
        #rec.draw(u,params,f,updates,actual_step,res,path+f'/NGD_{iteration}.png')
        losslist.append(l2_error)
        print(
            f'NGD Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error}'
            f' with stepsize: {actual_step}'
        )


with open(path+'losshist.pkl','wb') as pfile:
    pkl.dump(losslist,pfile)

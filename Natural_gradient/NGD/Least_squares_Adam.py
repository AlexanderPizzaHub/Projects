import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax 
import pickle as pkl
import optax
import os 
import matplotlib.pyplot as plt
from jax.numpy.linalg import lstsq
from utils import tools,model,validation

with open("dataset/5000points",'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)
    c_c = pkl.load(pfile)

print(d_c.shape,b_c.shape)


key = random.PRNGKey(0)
layers = [2,20,20,1]
params = model.init_network_params(key,layers)
u = model.NN(jnp.tanh)

g = tools.gram(u,params,d_c)
f = lambda x: (jnp.sin(jnp.pi*x[0])*jnp.sin(jnp.pi*x[1]))+jnp.sin(jnp.pi*3/2*x[0])*jnp.sin(jnp.pi*3/2*x[1])

residual = lambda params,x: 0.5*(u(params,x) - f(x))**2
v_residual = jit(vmap(residual,(None,0)))

@jit
def loss(params):
    return jnp.mean(v_residual(params,d_c))

rec = validation.recorder(Nv=64)
grid = jnp.linspace(0, 30, 31)
steps = (0.5**grid)*100.0
ls_update = tools.grid_line_search_factory(loss, steps)

exponential_decay = optax.exponential_decay(
    init_value=0.01, 
    transition_steps=1000,
    transition_begin=1000,
    decay_rate=0.1,
    end_value=0.0000001
)

optimizer = optax.adam(learning_rate=exponential_decay)
opt_state = optimizer.init(params)

losslist = []
# adam gradient descent with line search
for iteration in jnp.arange(0,2000):
    grads = grad(loss)(params)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if iteration % 10 == 0:
        # errors
        l2_error = jnp.sqrt(loss(params))
    
        print(
            f'Adam Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error}'
        )
        losslist.append(l2_error)

with open('results/losslist_adam_mlp.pkl','wb') as pfile:
    pkl.dump(losslist,pfile)
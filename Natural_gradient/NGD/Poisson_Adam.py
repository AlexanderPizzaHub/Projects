import jax.numpy as jnp
from jax import grad, jit, vmap,hessian
from jax import random
import jax 
import pickle as pkl
import optax
from utils import model,validation,pde,tools

with open("dataset/5000points",'rb') as pfile:
    d_c = pkl.load(pfile)
    b_c = pkl.load(pfile)
    c_c = pkl.load(pfile)

print(d_c.shape,b_c.shape)

key = random.PRNGKey(0)
layers = [2,20,10,10,1]
params = model.init_network_params(key,layers)
u = model.NN(jnp.tanh)

# solution
@jit
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x))

# rhs
@jit
def f(x):
    return 2. * jnp.pi**2 * u_star(x)

# compute residual
laplace_model = lambda params: pde.laplace(lambda x: u(params, x))
residual = lambda params, x: (laplace_model(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

lap_func = lambda params: pde.laplace(lambda x: u(params,x))

#v_lap = jit(vmap(lap_func,(None,0)))

residual_dom = lambda params,x: 0.5*(lap_func(params)(x) + f(x))**2
residual_bdry = lambda params,x: 0.5*(u(params,x)-0)**2 

v_residual_dom = jit(vmap(residual_dom,(None,0)))
v_residual_bdry = jit(vmap(residual_bdry,(None,0)))

@jit
def loss(params):
    return jnp.mean(v_residual_dom(params,d_c)) + 200 * jnp.mean(v_residual_bdry(params,b_c))


# errors
error = lambda x: u(params, x) - u_star(x)
rel_norm = jnp.mean(jnp.square(vmap(u_star)(d_c)))**0.5
print(rel_norm)
v_error = vmap(error) #Caution: this cannot apply jit because it involves a changing params, which is global variable

def l2_norm(f):
    return jnp.mean(lambda x: (f(x))**2)**0.5

# optimizer settings
exponential_decay = optax.exponential_decay(
    init_value=0.001, 
    transition_steps=2000,
    transition_begin=1500,
    decay_rate=0.1,
    end_value=0.0000001
)

optimizer = optax.adam(learning_rate=exponential_decay)
opt_state = optimizer.init(params)
   
losslist = []
# adam gradient descent with line search
for iteration in jnp.arange(0,4000):
    grads = grad(loss)(params)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if iteration % 10 == 0:
        # errors
        l2_error = jnp.mean(v_error(d_c)**2)**0.5/rel_norm
        losslist.append(l2_error)
        print(
            f'Adam Iteration: {iteration} with loss: {loss(params)} with error '
            f'L2: {l2_error}'
        )


with open('results/losslist_adam_poisson.pkl','wb') as pfile:
    pkl.dump(losslist,pfile)
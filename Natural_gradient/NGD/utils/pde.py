import jax.numpy as jnp
from jax import grad, jit, vmap,hessian
from jax import random
import jax 

def laplace(func):

    hess = hessian(func)

    lap = lambda x: jnp.trace(hess(x))

    return lap 

def parabolic(func):
    lap = laplace(func)
    par = lambda x: grad(func)(x[-1])+lap(x[:-1])

    return par 




"""
Implementation of Gramians and natural gradients.

"""
import jax.numpy as jnp
from jax import grad, vmap
import jax.flatten_util
from jax.numpy.linalg import lstsq


def pre_gram_factory(model, trafo):

    def del_theta_model(params, x):
        return grad(model)(params, x)
    
    def pre_gram(params, x):
        
        def g(y):
            return trafo(
                lambda z: model(params, z),
                lambda z: del_theta_model(params, z),
            )(y)
        
        flat = jax.flatten_util.ravel_pytree(g(x))[0]
        flat_col = jnp.reshape(flat, (len(flat), 1))
        flat_row = jnp.reshape(flat, (1, len(flat)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram

def gram_factory(model, trafo, integrator):

    pre_gram = pre_gram_factory(model, trafo)
    v_pre_gram = vmap(pre_gram, (None, 0))
    
    def gram(params):
        gram_matrix = integrator(lambda x: v_pre_gram(params, x))
        return gram_matrix
    
    return gram


def greedy(gram_matrix,nn_grad):
    h = jnp.zeros_like(nn_grad)
    index_list = jnp.array([],dtype=int)
    normreg=0.0001
    tv_norm = jnp.diag(gram_matrix)**0.5
    for i in range(30):
        residual = jnp.divide(nn_grad - jnp.matmul(gram_matrix,h),tv_norm+normreg)
        #residual = nn_grad - jnp.matmul(gram_matrix,h)
        residual_deleted = residual.at[index_list].set(0)
        new_ind = jnp.argmax(jnp.abs(residual_deleted))
        if not new_ind in index_list:
            index_list = jnp.append(index_list,new_ind)
            #if i==0:
            #    index_list = jnp.reshape(index_list,(1,-1))

        else:
            print('duplicate index appeared')

        #h_update = lstsq(gram_matrix[jnp.ix_(index_list,index_list)], nn_grad[index_list])[0]
        h_update = jnp.linalg.solve(gram_matrix[jnp.ix_(index_list,index_list)], nn_grad[index_list])
        #print(h_update)
        #h = (1-stepsize)*h + reg*stepsize* jnp.divide(residual[update_ind],tv_norm[update_ind]+normreg)*e
        #h.at[index_list].set(h_update)
        h = jnp.zeros_like(nn_grad).at[index_list].set(h_update)

    return h 

def nat_grad_factory(gram):

    def natural_gradient(params, tangent_params):

        gram_matrix = gram(params)
        flat_tangent, retriev_pytree  = jax.flatten_util.ravel_pytree(tangent_params)
        
        # solve gram dot flat_tangent.
        #flat_nat_grad = jnp.clip(lstsq(gram_matrix, flat_tangent)[0],-1000,1000)
        flat_nat_grad = jnp.linalg.solve(gram_matrix, flat_tangent)     #If not greedy, use this line
        #flat_nat_grad = jnp.clip(greedy(gram_matrix,flat_tangent),-1000,1000) # If greedy, use this line.

        # if gramian is zero then lstsq gives back nan...
        if jnp.isnan(flat_nat_grad[0]):
            return retriev_pytree(jnp.zeros_like(flat_nat_grad))

        else:
            return retriev_pytree(flat_nat_grad)

    return natural_gradient
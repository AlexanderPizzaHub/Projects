import jax.numpy as jnp
from jax import grad, jit, vmap,hessian
from jax import random
import jax 
from jax.numpy.linalg import lstsq

norm_reg = 0.0001

def pre_gram(model,params,x):
    ''' 
    value of gramian at given point x.
    '''
    d_theta_u = grad(model)(params,x)
    d_theta_u_flattened, unravel = jax.flatten_util.ravel_pytree(d_theta_u)
    f_col = jnp.reshape(d_theta_u_flattened,(-1,1))
    f_row = jnp.reshape(d_theta_u_flattened,(1,-1))
    return jnp.matmul(f_col,f_row)

    
def gram(model,params,d_c):
    #v_pre_gram = lambda x: pre_gram(model,params,x)
    gram_points = vmap(pre_gram,(None,None,0))(model,params,d_c)
    gram_int = jnp.mean(gram_points,axis=0)
    return gram_int




def preconditioner(gram_matrix,tangent_params,stepsize):
    ''' 
    generate the standard preconditioned gradient
    '''
    tp_flat, unravel = jax.flatten_util.ravel_pytree(tangent_params)
    prec_grad = -stepsize * lstsq(gram_matrix, tp_flat)[0]
    return unravel(prec_grad)


def greedy(n_steps,gram_matrix,nn_grad):

    h = jnp.zeros_like(nn_grad)
    index_list = jnp.array([],dtype=int)
    residual_norm_list = []
    tv_norm = jnp.sqrt(jnp.diag(gram_matrix))

    for i in range(n_steps):
        residual = jnp.divide(nn_grad - jnp.matmul(gram_matrix,h),tv_norm+norm_reg)
        #residual = nn_grad - jnp.matmul(gram_matrix,h)
        
        residual_deleted = residual.at[index_list].set(0)
        new_ind = jnp.argmax(jnp.abs(residual_deleted))

        index_list = jnp.append(index_list,new_ind)

        h_update = lstsq(gram_matrix[jnp.ix_(index_list,index_list)], nn_grad[index_list])[0]

        #h_update = jnp.linalg.solve(gram_matrix[jnp.ix_(index_list,index_list)], nn_grad[index_list])

        h = jnp.zeros_like(nn_grad).at[index_list].set(h_update)

        res = jnp.matmul(h.T,jnp.matmul(gram_matrix,h))-2*jnp.matmul(h.T,nn_grad)
        residual_norm_list.append(res)

    return h,residual_norm_list


def preconditioner_greedy(gram_matrix,tangent_params,greedy_steps,stepsize):
    tp_flat, unravel = jax.flatten_util.ravel_pytree(tangent_params)

    h,res = greedy(greedy_steps,gram_matrix,tp_flat)
    prec_grad = stepsize * h

    return unravel(prec_grad),res



def grid_line_search_factory(loss, steps):
    
    def loss_at_step(step, params, tangent_params):
        updated_params = [(w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, tangent_params)]
        return loss(updated_params)
        
    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None)))    

    @jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, tangent_params)], step_size
    
    return grid_line_search_update


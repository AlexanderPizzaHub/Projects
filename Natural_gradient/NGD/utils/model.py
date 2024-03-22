import jax.numpy as jnp
import jax


initializer = jax.nn.initializers.glorot_uniform()
def random_layer_params(key,m,n,scale=1):
    return scale * initializer(key, (n,m),dtype=jnp.float64), jnp.zeros((n,))

def init_network_params(key,sizes):
    return [random_layer_params(key,x,y) for x,y in zip(sizes[:-1],sizes[1:])]

#params: [[[L1,L2],[L1,]],[[L2,L3],[L2,]],...,[[Ln-1,Ln],[Ln-1,]]]
#        list -> tuple -> array
#        layers -> weight, bias -> coefficients

def NN(activation):
    def model(params,x):
        output = x 
        for w,b in params[:-1]:
            linear = jnp.dot(output,w.T) + b
            output = activation(linear)
        w,b = params[-1]
        output = jnp.reshape(jnp.dot(output,w.T) + b,())
        return output
    return model 
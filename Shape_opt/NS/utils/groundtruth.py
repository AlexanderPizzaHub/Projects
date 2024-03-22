import numpy as np
from sympy import *

alpha = 0.01

x1,x2 = symbols('x1 x2')

f = 2.5*(x1 + 0.4 - x2**2)**2 + x1**2 + x2**2 -1

variables = [x1,x2]

ldf = lambdify(variables,f,'numpy')

def from_seq_to_array(items):
    out = list()
    for item in items:
        out.append(np.array(item).reshape(-1,1))
    
    if len(out)==1:
        out = out[0]
    return out

def data_gen_interior(collocations):
    #how to parse the input?
    f = [
        ldf(d[0],d[1]) for d in collocations
    ]

    return from_seq_to_array([f])[0]

import numpy as np
from sympy import *

alpha = 0.01

x1,x2 = symbols('x1 x2')

low = -0.5
high = 0.7

r = sqrt((x1)**2+(x2)**2)
sintheta = (x2)/r

y = r**2
p = alpha*(r-1)*(r-3)*sintheta

variables = [x1,x2]

laplacian_y = 0
for x in variables:
    laplacian_y += diff(y,x,x)

laplacian_p = 0
for x in variables:
    laplacian_p += diff(p,x,x)

u = Min(high,Max(-p/alpha,low))  

y_dat = y + laplacian_p
f = - laplacian_y - u

#Generates all the ground truth data needed. 

#boundary is periodic, not needed.
ldy = lambdify(variables,y,'numpy')
ldu = lambdify(variables,u,'numpy')
ldp = lambdify(variables,p,'numpy')
ldydat = lambdify(variables,y_dat,'numpy')
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
    y_gt = [
        ldy(d[0],d[1]) for d in collocations
    ]
        
    u_gt = [
        ldu(d[0],d[1]) for d in collocations
    ]

    p_gt = [
        ldp(d[0],d[1]) for d in collocations
    ]

    y_data = [
        ldydat(d[0],d[1]) for d in collocations
    ]
    
    f = [
        ldf(d[0],d[1]) for d in collocations
    ]

    return from_seq_to_array([y_gt,u_gt,p_gt,y_data,f])

def data_gen_bdry(collocations):
    #how to parse the input?
    y_gt = [
        ldy(d[0],d[1]) for d in collocations
    ]

    return from_seq_to_array([y_gt])
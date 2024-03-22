import numpy as np
from sympy import *

alpha = 0.01

x1,x2 = symbols('x1 x2')
y = exp(x1*(1-x1))*sin(pi*x2) + exp(x2*(1-x2))*sin(pi*x1)
p = x1*(1+cos(pi*x1))*x2*(1+cos(pi*x2))

d = Piecewise((y**3,And(And(x1<0.75,x1>0.25),And(x2>0.25,x2<0.75))),(3.0*y**3,True))

dp = Piecewise((3*y**2,And(And(x1<0.75,x1>0.25),And(x2>0.25,x2<0.75))),(9.0*y**2,True))

variables = [x1,x2]

laplacian_y = 0
for x in variables:
    laplacian_y += diff(y,x,x)

laplacian_p = 0
for x in variables:
    laplacian_p += diff(p,x,x)

u = - p/alpha

y_dat =  y + laplacian_p - p - dp * p 
f = - laplacian_y + y + d - u

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
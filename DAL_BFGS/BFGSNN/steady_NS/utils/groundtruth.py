import numpy as np
from sympy import *

alpha = 0.01
mu = 1.0

x1,x2 = symbols('x1 x2')

variables = [x1,x2]

y1 = exp(-0.05*mu)*sin(pi*x1)**2*sin(pi*x2)*cos(pi*x2)
y2 = exp(-0.05*mu)*(-sin(pi*x2)**2)*sin(pi*x1)*cos(pi*x1)

ld1 = (exp(-0.05*mu)-exp(-mu)) * (sin(pi*x1)**2*sin(pi*x2)*cos(pi*x2))
ld2 = (exp(-0.05*mu)-exp(-mu)) * (-sin(pi*x2)**2*sin(pi*x1)*cos(pi*x1))

p = 0
v = 0

u1 = ld1 
u2 = ld2

def laplacian(inp):
    lap = 0
    for x in variables:
        lap += diff(inp,x,x)

    return lap


f1 = - mu * laplacian(y1) - u1 + y1*diff(y1,x1) + y2*diff(y1,x2)
f2 = - mu * laplacian(y2) - u2 + y1*diff(y2,x1) + y2*diff(y2,x2)

yd1 = y1 - ( -mu*laplacian(ld1)+ y1*diff(ld1,x1)+y2*diff(ld1,x2) - diff(y1,x1)*ld1 - diff(y1,x2)*ld2)
yd2 = y2 - ( -mu*laplacian(ld2)+ y1*diff(ld2,x1)+y2*diff(ld2,x2) - diff(y2,x1)*ld1 - diff(y2,x2)*ld2)


#Generates all the ground truth data needed. 

#boundary is periodic, not needed.
ldy = lambdify(variables,[y1,y2],'numpy')
ldu = lambdify(variables,[u1,u2],'numpy')
ldp = lambdify(variables,p,'numpy')
ldydat = lambdify(variables,[yd1,yd2],'numpy')
ldf = lambdify(variables,[f1,f2],'numpy')

def from_seq_to_array(items):
    out = list()
    for item in items:
        out.append(np.array(item))
    
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
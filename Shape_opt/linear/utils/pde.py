import torch
import numpy as np
#Function to compute pde term

mse_loss = torch.nn.MSELoss()
#use x^2 to verify the laplacian

def pde(x1,x2,net):
    out = net(x1,x2)

    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1.sum(),x1,create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    return -u_xx1 - u_xx2

def adjoint(x1,x2,net):
    out = net(x1,x2)

    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1.sum(),x1,create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    return -u_xx1 - u_xx2

def reg(x1,x2,net):
    out = net(x1,x2)
    u1_x1 = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    u1_xx1 = torch.autograd.grad(u1_x1.sum(),x1,create_graph=True)[0]

    u1_x2 = torch.autograd.grad(out[:,0].sum(),x2,create_graph=True)[0]
    u1_xx2 = torch.autograd.grad(u1_x2.sum(),x2,create_graph=True)[0]

    u2_x1 = torch.autograd.grad(out[:,1].sum(),x1,create_graph=True)[0]
    u2_xx1 = torch.autograd.grad(u2_x1.sum(),x1,create_graph=True)[0]

    u2_x2 = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]
    u2_xx2 = torch.autograd.grad(u2_x2.sum(),x2,create_graph=True)[0]

    return torch.concatenate([-u1_xx1-u1_xx2,-u2_xx1-u2_xx2],dim=1) + out

#Function to compute the bdry
def bdry(bx1,bx2,net):
    out = net(bx1,bx2)
    return out


#The loss
def pdeloss(net,px1,px2,pdedata,bx1,bx2,bdrydata,bw):
    
    #pdedata is f.
    pout = pde(px1,px2,net)
    
    bout = bdry(bx1,bx2,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    loss = pres + bw*bres
    return loss,[pres,bres],[pout-pdedata,bout-bdrydata]

def adjloss(net,px1,px2,pdedata,bx1,bx2,bdrydata,bw):
    
    #pdedata is f.
    pout = adjoint(px1,px2,net)
    
    bout = bdry(bx1,bx2,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    loss = pres + bw*bres
    return loss,pres,bres

'''def compute_normal(phi,bx1,bx2,original_normal=None):
    #computes the normal vector after deformation by phi. 
    #NEED TO CHECK VALID
    #Original shape is assumes to be circle. Hence orginal_normal is omitted.
    phi_out = phi(bx1,bx2)
    phi1x = torch.autograd.grad(phi_out[:,0].sum(),bx1,create_graph=True)[0]
    phi1y = torch.autograd.grad(phi_out[:,1].sum(),bx2,create_graph=True)[0]
    phi2x = torch.autograd.grad(phi_out[:,0].sum(),bx1,create_graph=True)[0]
    phi2y = torch.autograd.grad(phi_out[:,1].sum(),bx2,create_graph=True)[0]
    tangent_vec = (bx1*phi1x+bx2*phi2x,bx1*phi1y+bx2*phi2y)
    normal_vec = torch.concatenate([tangent_vec[1],-tangent_vec[0]],dim=1)/torch.sqrt(tangent_vec[0]**2+tangent_vec[1]**2)

    return normal_vec'''

counter_dir = torch.tensor([-1,1])
def compute_normal(bd_col):
    #collocations must be counter-clockwise
    indexing = np.arange(len(bd_col))
    normal = torch.zeros(len(bd_col),2)
    normal = counter_dir * torch.roll((bd_col[np.roll(indexing,1)] - bd_col[np.roll(indexing,-1)]),dims=-1,shifts=1)
    normal = torch.div(normal,torch.linalg.norm(normal,dim=1).unsqueeze(-1))
    return normal


def bdry_flux(bx1,bx2,normal,net):  #normal:[N,2,1]
    out = net(bx1,bx2)
    u_x1 = torch.autograd.grad(out.sum(),bx1,create_graph=True)[0]
    u_x2 = torch.autograd.grad(out.sum(),bx2,create_graph=True)[0]
    return u_x1*normal[:,0] + u_x2*normal[:,1]

def bdry_misfit(net,y,p,bx1,bx2,normal):
    phi = net(bx1,bx2) #[N,d]
    normal = normal.unsqueeze(-1)
    #jac_phi = torch.autograd.functional.jacobian(net,(bx1,bx2)) #check shape 
    phi1x = torch.autograd.grad(phi[:,0].sum(),bx1,create_graph=True)[0]
    phi1y = torch.autograd.grad(phi[:,0].sum(),bx2,create_graph=True)[0]
    phi2x = torch.autograd.grad(phi[:,1].sum(),bx1,create_graph=True)[0]
    phi2y = torch.autograd.grad(phi[:,1].sum(),bx2,create_graph=True)[0]
    partial_phi_n = torch.concatenate([phi1x*normal[:,0]+phi1y*normal[:,1] , phi2x*normal[:,0]+phi2y*normal[:,1]],dim=1)

    partial_y = bdry_flux(bx1,bx2,normal,y)
    partial_p = bdry_flux(bx1,bx2,normal,p)

    #print(partial_y.shape)
    #print((torch.mul(partial_y*partial_p,normal.squeeze(-1))).shape,partial_phi_n.shape)
    misfit = partial_phi_n + torch.mul(partial_y*partial_p,normal.squeeze(-1))

    return misfit 


def regloss(net,y,p,dx1,dx2,pdedata,bx1,bx2,bw,normal):
    pout = reg(dx1,dx2,net)
    
    bmisfit = bdry_misfit(net,y,p,bx1,bx2,normal)
    
    pres = mse_loss(pout,pdedata)
    bres = torch.mean(torch.square(bmisfit))
    loss = pres + bw*bres

    return loss,pres,bres

def cost_mse(y,data,u,ld,cx1,cx2):
    yout = y(cx1,cx2)
    if not isinstance(u,torch.Tensor):
        uout = u(cx1,cx2)
    else:
        uout = u
    misfit = 0.5 *torch.square(yout-data) + ld * 0.5 * torch.square(uout)
    cost = torch.mean(misfit)
    return cost


#check normal vector is correct!
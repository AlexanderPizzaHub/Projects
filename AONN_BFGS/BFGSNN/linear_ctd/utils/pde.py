import torch

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

def cost_mse(y,data,u,ld,cx1,cx2):
    yout = y(cx1,cx2)
    if not isinstance(u,torch.Tensor):
        uout = u(cx1,cx2)
    else:
        uout = u
    misfit = 0.5 *torch.square(yout-data) + ld * 0.5 * torch.square(uout)
    cost = torch.mean(misfit)
    return cost

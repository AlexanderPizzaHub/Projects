import torch

#Function to compute pde term

mse_loss = torch.nn.MSELoss()
#use x^2 to verify the laplacian

def dist(labelx1,labelx2,out):
    d = torch.zeros(out.shape)
    d[torch.logical_and(labelx1<=0, labelx2<=0)] = 1.0 * torch.pow(out[torch.logical_and(labelx1<=0, labelx2<=0)],3)
    d[torch.logical_or(labelx1>0, labelx2>0)] = 3.0 * torch.pow(out[torch.logical_or(labelx1>0, labelx2>0)],3)
    
    return d

def distp(labelx1,labelx2,out_primal,out_adjoint):
    d = torch.zeros(out_primal.shape)
    d[torch.logical_and(labelx1<=0, labelx2<=0)] = 3 * 1.0 * torch.pow(out_primal[torch.logical_and(labelx1<=0, labelx2<=0)],2) * out_adjoint[torch.logical_and(labelx1<=0, labelx2<=0)]
    d[torch.logical_or(labelx1>0, labelx2>0)] = 3 * 3.0 * torch.pow(out_primal[torch.logical_or(labelx1>0, labelx2>0)],2) * out_adjoint[torch.logical_or(labelx1>0, labelx2>0)]
    return d

def pde(x1,x2,net,labelx1,labelx2):
    out = net(x1,x2)

    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1.sum(),x1,create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    return -u_xx1 - u_xx2 + out + dist(labelx1,labelx2,out)

def adjoint(x1,x2,net,primal_state_value,labelx1,labelx2):
    out = net(x1,x2)

    u_x1 = torch.autograd.grad(out.sum(),x1,create_graph=True)[0]
    u_xx1 = torch.autograd.grad(u_x1.sum(),x1,create_graph=True)[0]

    u_x2 = torch.autograd.grad(out.sum(),x2,create_graph=True)[0]
    u_xx2 = torch.autograd.grad(u_x2.sum(),x2,create_graph=True)[0]

    return -u_xx1 - u_xx2 + out + distp(labelx1,labelx2,primal_state_value,out)

#Function to compute the bdry
def bdry(bx1,bx2,net):
    out = net(bx1,bx2)
    return out


#The loss
def pdeloss(net,px1,px2,labelx1,labelx2,pdedata,bx1,bx2,bdrydata,bw):
    
    #pdedata is f.
    pout = pde(px1,px2,net,labelx1,labelx2)
    
    bout = bdry(bx1,bx2,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    loss = pres + bw*bres
    return loss,[pres,bres],[pout-pdedata,bout-bdrydata]

def adjloss(net,primal_net,px1,px2,labelx1,labelx2,pdedata,bx1,bx2,bdrydata,bw):

    pout = adjoint(px1,px2,net,primal_net(px1,px2),labelx1,labelx2)
    
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
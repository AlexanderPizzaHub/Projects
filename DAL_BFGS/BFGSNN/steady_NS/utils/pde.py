import torch

#Function to compute pde term

mse_loss = torch.nn.MSELoss()
#use x^2 to verify the laplacian



def red_out(x1,x2,net):
    out = net(x1,x2)
    return torch.sum(out,dim=0)

class redfunc(object):
    def __init__(self,net):
        self.net = net 

    def __call__(self,x1,x2):
        return red_out(x1,x2,self.net)

def pde_old(x1,x2,net,red_func,p,mu): #THIS JACOBIAN IS WRONG
    pout = p(x1,x2)
    jac = torch.autograd.functional.jacobian(red_func,(x1,x2),create_graph=True)
    '''
    jac[0][0] -> u1_x1
    jac[0][1] -> u1_x2
    jac[1][0] -> u2_x1 
    jac[1][1] -> u2_x2
    '''
     
    #OPTIMIZABLE
    u1_x1x1 = torch.autograd.grad(jac[0][0].sum(),x1,create_graph=True)[0] 
    u1_x2x2 = torch.autograd.grad(jac[0][1].sum(),x2,create_graph=True)[0]

    u2_x1x1 = torch.autograd.grad(jac[1][0].sum(),x1,create_graph=True)[0]
    u2_x2x2 = torch.autograd.grad(jac[1][1].sum(),x2,create_graph=True)[0]

    p_x1 = torch.autograd.grad(pout.sum(),x1,create_graph=True)[0]
    p_x2 = torch.autograd.grad(pout.sum(),x2,create_graph=True)[0]

    out = net(x1,x2).unsqueeze(-1)
    pde_1 = - (u1_x1x1 + u1_x2x2) + p_x1 + out[:,0]*jac[0][0] + out[:,1]*jac[0][1]
    pde_2 = - (u2_x1x1 + u2_x2x2) + p_x2 + out[:,0]*jac[1][0] + out[:,1]*jac[1][1]

    return torch.concat([pde_1,pde_2],dim=1), [jac[0][0],jac[1][1]] #divergence term

def pde_backup(x1,x2,net,p,mu):
    pout = p(x1,x2)
    out = net(x1,x2).unsqueeze(-1)
    #jac = torch.autograd.functional.jacobian(red_func,(x1,x2),create_graph=True)
    u1_x1 = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    u1_x2 = torch.autograd.grad(out[:,0].sum(),x2,create_graph=True)[0]
    u2_x1 = torch.autograd.grad(out[:,1].sum(),x1,create_graph=True)[0]
    u2_x2 = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]
    '''
    jac[0][0] -> u1_x1
    jac[0][1] -> u1_x2
    jac[1][0] -> u2_x1 
    jac[1][1] -> u2_x2
    '''
     
     #OPTIMIZABLE
    u1_x1x1 = torch.autograd.grad(u1_x1.sum(),x1,create_graph=True)[0] 
    u1_x2x2 = torch.autograd.grad(u1_x2.sum(),x2,create_graph=True)[0]

    u2_x1x1 = torch.autograd.grad(u2_x1.sum(),x1,create_graph=True)[0]
    u2_x2x2 = torch.autograd.grad(u2_x2.sum(),x2,create_graph=True)[0]

    p_x1 = torch.autograd.grad(pout.sum(),x1,create_graph=True)[0]
    p_x2 = torch.autograd.grad(pout.sum(),x2,create_graph=True)[0]

    
    pde_1 = - mu*(u1_x1x1 + u1_x2x2) + p_x1 + out[:,0]*u1_x1 + out[:,1]*u1_x2
    pde_2 = - mu*(u2_x1x1 + u2_x2x2) + p_x2 + out[:,0]*u2_x1 + out[:,1]*u2_x2

    return torch.concat([pde_1,pde_2],dim=1), u1_x1+u2_x2








def grad(out,x1,x2):
    #jacobian
    grad11 = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    grad12 = torch.autograd.grad(out[:,0].sum(),x2,create_graph=True)[0]
    grad21 = torch.autograd.grad(out[:,1].sum(),x1,create_graph=True)[0]
    grad22 = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]

    
    grad1 = torch.concatenate([grad11,grad12],dim=1).unsqueeze(1)
    grad2 = torch.concatenate([grad21,grad22],dim=1).unsqueeze(1)
    jac = torch.concatenate([grad1,grad2],dim=1)
    return jac #[N,2,2]

def div_mat(out,x1,x2):
    #out:[N,2,2]
    #divergence
    div11 = torch.autograd.grad(out[:,0,0].sum(),x1,create_graph=True)[0]
    div12 = torch.autograd.grad(out[:,0,1].sum(),x1,create_graph=True)[0]
    div21 = torch.autograd.grad(out[:,1,0].sum(),x2,create_graph=True)[0]
    div22 = torch.autograd.grad(out[:,1,1].sum(),x2,create_graph=True)[0]

    div1 = torch.concatenate([div11,div12],dim=1)
    div2 = torch.concatenate([div21,div22],dim=1)
    return div1+div2 #[N,2]

def div_vec(out,x1,x2):
    #out:[N,2]
    div1 = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    div2 = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]
    return div1+div2 #[N,1]




def pde(x1,x2,net,p,mu):
    u_out = net(x1,x2) #[N,2]
    pout = p(x1,x2)

    p_x1 = torch.autograd.grad(pout.sum(),x1,create_graph=True)[0]
    p_x2 = torch.autograd.grad(pout.sum(),x2,create_graph=True)[0]


    jac_u = grad(u_out,x1,x2) #[N,2,2]
    term_1 = -mu * div_mat(torch.transpose(jac_u,1,2),x1,x2)
    #term_1 = -mu * div_mat(jac_u,x1,x2)
    
    term_2 =  torch.matmul(jac_u,u_out.unsqueeze(-1)).squeeze(-1)

    term_3 = torch.concatenate([p_x1,p_x2],dim=1)


    lhs = term_1 + term_2 + term_3

    div_u = div_vec(u_out,x1,x2) #[N,1]
    return lhs,div_u









def adjoint(x1,x2,net,v,primal_net,mu):
    out = net(x1,x2).unsqueeze(-1)

    u1_x1 = torch.autograd.grad(out[:,0].sum(),x1,create_graph=True)[0]
    u1_x2 = torch.autograd.grad(out[:,0].sum(),x2,create_graph=True)[0]
    u2_x1 = torch.autograd.grad(out[:,1].sum(),x1,create_graph=True)[0]
    u2_x2 = torch.autograd.grad(out[:,1].sum(),x2,create_graph=True)[0]

    u1_x1x1 = torch.autograd.grad(u1_x1.sum(),x1,create_graph=True)[0] 
    u1_x2x2 = torch.autograd.grad(u1_x2.sum(),x2,create_graph=True)[0]

    u2_x1x1 = torch.autograd.grad(u2_x1.sum(),x1,create_graph=True)[0]
    u2_x2x2 = torch.autograd.grad(u2_x2.sum(),x2,create_graph=True)[0]

    vout = v(x1,x2)
    v_x1 = torch.autograd.grad(vout.sum(),x1,create_graph=True)[0]
    v_x2 = torch.autograd.grad(vout.sum(),x2,create_graph=True)[0]

    primal_out = primal_net(x1,x2).unsqueeze(-1)
    p1_x1 = torch.autograd.grad(primal_out[:,0].sum(),x1,create_graph=True)[0]
    p1_x2 = torch.autograd.grad(primal_out[:,0].sum(),x2,create_graph=True)[0]
    p2_x1 = torch.autograd.grad(primal_out[:,1].sum(),x1,create_graph=True)[0]
    p2_x2 = torch.autograd.grad(primal_out[:,1].sum(),x2,create_graph=True)[0]

    pde1 = -mu * (u1_x1x1 + u1_x2x2) - (primal_out[:,0]*u1_x1 + primal_out[:,1]*u1_x2) + (p1_x1*out[:,0]+p1_x2*out[:,1]) + v_x1
    pde2 = -mu * (u2_x1x1 + u2_x2x2) - (primal_out[:,0]*u2_x1 + primal_out[:,1]*u2_x2) + (p2_x1*out[:,0]+p2_x2*out[:,1]) + v_x2

    return torch.concat([pde1,pde2],dim=1),[u1_x1,u2_x2]

#Function to compute the bdry
def bdry(bx1,bx2,net):
    out = net(bx1,bx2)
    return out

def init(ix1,ix2,net):
    out = net(ix1,ix2)
    return out

def bdry_flux(bx1,bx2,bdrymark,net):
    out = net(bx1,bx2)
    u_x1 = torch.autograd.grad(out.sum(),bx1,create_graph=False)[0]
    u_x2 = torch.autograd.grad(out.sum(),bx2,create_graph=False)[0]
    return u_x1*bdrymark[:,0] + u_x2*bdrymark[:,1]

#The loss
def pdeloss(net,p,px1,px2,pdedata,bx1,bx2,bdrydata,bw,divdata,dw,red_func,mu):
    
    #pdedata is f.
    pout,div = pde(px1,px2,net,p,mu)
    
    bout = bdry(bx1,bx2,net)
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)

    dres = mse_loss(div,divdata)
    loss = pres + bw*bres + dw*dres

    return loss,[pres,bres,dres],[pout-pdedata,bout-bdrydata,div]

def adjloss(net,v,primal_net,px1,px2,pdedata,bx1,bx2,bdrydata,bw,divdata,dw,mu):
    
    #pdedata is f.
    pout,div = adjoint(px1,px2,net,v,primal_net,mu)
    
    bout = bdry(bx1,bx2,net)
    
    pres = mse_loss(pout,pdedata)
    bres = mse_loss(bout,bdrydata)
    divsum = div[0]+div[1]
    dres = mse_loss(divsum,divdata)
    loss = pres + bw*bres + dw*dres
    return loss,pres,bres,dres

def cost_mse(y,data,u,ld,cx1,cx2):
    yout = y(cx1,cx2)
    if not isinstance(u,torch.Tensor):
        uout = u(cx1,cx2)
    else:
        uout = u
    misfit = 0.5 *torch.square(yout-data) + ld * 0.5 * torch.square(uout)
    cost = torch.mean(misfit)
    return cost

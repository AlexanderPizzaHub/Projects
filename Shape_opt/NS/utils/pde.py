import torch
import numpy as np
#Function to compute pde term

mse_loss = torch.nn.MSELoss()



counter_dir = torch.tensor([-1,1])
def compute_normal(bd_col):
    #collocations must be counter-clockwise
    #only for closed curves
    indexing = np.arange(len(bd_col))
    normal = torch.zeros(len(bd_col),2)
    normal = counter_dir * torch.roll((bd_col[np.roll(indexing,1)] - bd_col[np.roll(indexing,-1)]),dims=-1,shifts=1)
    normal = torch.div(normal,torch.linalg.norm(normal,dim=1).unsqueeze(-1))
    return normal.unsqueeze(-1)

def compute_normal_nonclosed(bd_col):
    #collocations must be counter-clockwise
    #for non-closed curves
    indexing = np.arange(len(bd_col))
    normal = torch.zeros(len(bd_col),2)
    normal = counter_dir * torch.roll((bd_col[np.roll(indexing,1)] - bd_col[np.roll(indexing,-1)]),dims=-1,shifts=1)
    normal[0,:] = counter_dir * torch.roll((bd_col[0,:] - bd_col[1,:]),dims=-1,shifts=1)
    normal[-1,:] = counter_dir * torch.roll((bd_col[-2,:] - bd_col[-1,:]),dims=-1,shifts=1)

    normal = torch.div(normal,torch.linalg.norm(normal,dim=1).unsqueeze(-1))
    return normal.unsqueeze(-1)

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
















def pde(u,p,nu,x1,x2): #checked correct
    u_out = u(x1,x2)
    p_out = p(x1,x2)

    p_x1 = torch.autograd.grad(p_out.sum(),x1,create_graph=True)[0]
    p_x2 = torch.autograd.grad(p_out.sum(),x2,create_graph=True)[0]

    term_1 = -torch.concatenate([p_x1,p_x2],dim=1)

    jac_u = grad(u_out,x1,x2)
    term_2 =  -torch.matmul(jac_u,u_out.unsqueeze(-1)).squeeze(-1)

    term_3 = nu * div_mat(torch.transpose(jac_u,1,2),x1,x2)

    lhs = term_1 + term_2 + term_3

    div_u = div_vec(u_out,x1,x2)

    return lhs,div_u

def bdry(net,p,nu,bx_o_list,normal_mapping_o):
    #items in list: bx_o_list = [gamma_i,gamma_w1,gamma_o,gamma_w2,gamma_f,gamma_w3],
    # each object in the list is a list containing [x1,x2].
    # corresponding normal vectors are contained in normal_mapping.
    
    ''' GAMMA I '''
    y_i = net(bx_o_list[0][0],bx_o_list[0][1])

    ''' GAMMA W '''
    y_w1 = net(bx_o_list[1][0],bx_o_list[1][1])
    y_w2 = net(bx_o_list[3][0],bx_o_list[3][1])    
    y_w3 = net(bx_o_list[5][0],bx_o_list[5][1])

    ''' GAMMA O '''
    p_o = p(bx_o_list[2][0],bx_o_list[2][1])
    u_o = net(bx_o_list[2][0],bx_o_list[2][1])

    bdry_o = p_o*(normal_mapping_o[2].squeeze(-1)) - nu*torch.matmul(grad(u_o,bx_o_list[2][0],bx_o_list[2][1]),normal_mapping_o[2]).squeeze(-1)

    

    ''' GAMMA F'''
    y_f = net(bx_o_list[4][0],bx_o_list[4][1])

    return [y_i,y_w1,bdry_o,y_w2,y_f,y_w3]



'''

State PDE checked correct.

'''




def adjoint(ld,q,primal_net,nu,x1,x2):
    #primal_net comes as neural networks.
    ldout = ld(x1,x2)
    qout = q(x1,x2)
    primal_net_out = primal_net(x1,x2)

    jac_ld = grad(ldout,x1,x2)

    term_1 = - nu * div_mat(torch.transpose(jac_ld,1,2),x1,x2)

    term_2 = - torch.matmul(jac_ld,primal_net_out.unsqueeze(-1)).squeeze(-1) #check sign

    jac_y = grad(primal_net_out,x1,x2)
    term_3 = torch.matmul(torch.transpose(jac_y,1,2),ldout.unsqueeze(-1)).squeeze(-1) #check sign

    q_x1 = torch.autograd.grad(qout.sum(),x1,create_graph=True)[0]
    q_x2 = torch.autograd.grad(qout.sum(),x2,create_graph=True)[0]

    term_4 = torch.concatenate([q_x1,q_x2],dim=1)

    div_ld = div_vec(ldout,x1,x2)

    lhs = term_1+term_2+term_3+term_4

    return lhs,div_ld



def bdry_adjoint(ld,q,primal,nu,bx_o_list,normal_mapping_o):
    #items in list: bx_o_list = [gamma_i,gamma_w1,gamma_o,gamma_w2,gamma_f,gamma_w3],
    # each object in the list is a list containing [x1,x2].
    # corresponding normal vectors are contained in normal_mapping.
    
    ''' GAMMA I '''
    y_i = ld(bx_o_list[0][0],bx_o_list[0][1])

    ''' GAMMA W '''
    y_w1 = ld(bx_o_list[1][0],bx_o_list[1][1])
    y_w2 = ld(bx_o_list[3][0],bx_o_list[3][1])    
    y_w3 = ld(bx_o_list[5][0],bx_o_list[5][1])

    ''' GAMMA O '''
    q_o = q(bx_o_list[2][0],bx_o_list[2][1])
    y_o = ld(bx_o_list[2][0],bx_o_list[2][1]) #[N,2]
    primal_o = primal(bx_o_list[2][0],bx_o_list[2][1]) #[N,2]

    #normal: [N,2,1]
    bdry_o = q_o*(normal_mapping_o[2].squeeze(-1)) - nu*torch.matmul(grad(y_o,bx_o_list[2][0],bx_o_list[2][1]),normal_mapping_o[2]).squeeze(-1) - torch.matmul(primal_o.unsqueeze(1),normal_mapping_o[2]).squeeze(-1) * y_o

    ''' GAMMA F'''
    y_f = ld(bx_o_list[4][0],bx_o_list[4][1])

    return [y_i,y_w1,bdry_o,y_w2,y_f,y_w3]














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


def bdry_flux(bx1,bx2,normal,out):  #normal:[N,2,1]
    u_n = torch.matmul(grad(out,bx1,bx2),normal).squeeze(-1)
    return u_n 

def compute_gradient(y,ld,yd,nu,bx1,bx2,normal):
    yout = y(bx1,bx2)
    ldout = ld(bx1,bx2)

    y_misfit = yout-yd
    term_1 = 0.5 * (y_misfit[:,0]**2 + y_misfit[:,1]**2).unsqueeze(-1) #[Nb,1]

    partial_y = bdry_flux(bx1,bx2,normal,yout) #[N,2]
    partial_ld = bdry_flux(bx1,bx2,normal,ldout)

    term_2 =  nu * (partial_y[:,0]*partial_ld[:,0]+partial_y[:,1]*partial_ld[:,1]).unsqueeze(-1)

    return term_1 + term_2 #[Nb,1]



def bdry_misfit(net,gradient,bx_o_list,normal_o):
    #[gamma_i,gamma_w1,gamma_o,gamma_w2,gamma_f,gamma_w3]

    ''' GAMMA I '''
    phi_i = net(bx_o_list[0][0],bx_o_list[0][1])

    ''' GAMMA W '''
    phi_w1 = net(bx_o_list[1][0],bx_o_list[1][1])
    phi_w2 = net(bx_o_list[3][0],bx_o_list[3][1])    
    phi_w3 = net(bx_o_list[5][0],bx_o_list[5][1])

    ''' GAMMA O '''
    phi_o = net(bx_o_list[2][0],bx_o_list[2][1])

    ''' GAMMA F'''
    phi_f = net(bx_o_list[4][0],bx_o_list[4][1])

    #jac_phi = torch.autograd.functional.jacobian(net,(bx1,bx2)) #check shape 
    phi1x = torch.autograd.grad(phi_f[:,0].sum(),bx_o_list[4][0],create_graph=True)[0]
    phi1y = torch.autograd.grad(phi_f[:,0].sum(),bx_o_list[4][1],create_graph=True)[0]
    phi2x = torch.autograd.grad(phi_f[:,1].sum(),bx_o_list[4][0],create_graph=True)[0]
    phi2y = torch.autograd.grad(phi_f[:,1].sum(),bx_o_list[4][1],create_graph=True)[0]
    partial_phi_n = torch.concatenate([phi1x*normal_o[4][:,0]+phi1y*normal_o[4][:,1] , phi2x*normal_o[4][:,0]+phi2y*normal_o[4][:,1]],dim=1)

    misfit_f = partial_phi_n + torch.matmul(normal_o[4],gradient.unsqueeze(-1)).squeeze(-1)
    #print(partial_phi_n.shape, torch.matmul(normal_o[4],gradient.unsqueeze(-1)).squeeze(-1).shape)
    return [phi_i,phi_w1,phi_o,phi_w2,misfit_f,phi_w3]

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
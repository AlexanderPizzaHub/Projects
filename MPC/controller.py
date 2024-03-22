import numpy as np
from scipy import optimize

class controller(object):
    def __init__(self):
        self.A = None 
        self.B = None 
        self.f = None 

        self.pred_horizon= None 
        self.control_horizon = None

        self.Q = None 
        self.R = None 
        self.F = None 

        self.f = None

        self.cons = None

    def setup_model(self,f):
        self.f = f

    def setup_constraints(self,A,b):
        ''' 
        Constraints in standard form: Ax-b >=0 
        '''
        self.cons = {'type':'ineq',
        'fun':lambda x: np.dot(A,x)-b,
        'jac':lambda x: A}

    def update_dynamics(self,A, x_lin, B, u_lin, f0):
        ''' 
        linearized dynamics: x_{t+1} = x_t + dt * (A x_t + B u_t + f0)
        x_lin and u_lin denote the point where the system is linearized. 
        '''
        self.A = A 
        self.B = B 
        self.f0 = f0
        self.x_lin = x_lin
        self.u_lin = u_lin

    def set_cost(self, Q, R, F):
        self.Q = Q 
        self.R = R 
        self.F = F

    def discretize_by_R(self,dt,R_tilde):
        self.dt = dt

        G = np.zeros([self.B.shape[0]*self.pred_horizon,self.B.shape[1]*self.pred_horizon])


        tmp_B = dt * self.B 

        del_A = np.eye(self.B.shape[0]) + dt * self.A 

        self.del_A_pow = [del_A]

        for i in range(0,self.pred_horizon):
            for j in range(0,self.pred_horizon-i):
                G[(j+i)*self.B.shape[0]:(j+i+1)*self.B.shape[0],(j)*self.B.shape[1]:(j+1)*self.B.shape[1]] = tmp_B 
            
            self.del_A_pow.append(del_A @ self.del_A_pow[-1])
            tmp_B = del_A @ tmp_B

        Q_tilde = np.zeros([self.pred_horizon*self.A.shape[0],self.pred_horizon*self.A.shape[1]])

        Q_tilde[0:(self.pred_horizon-1)*self.A.shape[0],0:(self.pred_horizon-1)*self.A.shape[1]] = np.kron(np.eye(self.pred_horizon-1),self.Q)

        Q_tilde[(self.pred_horizon-1)*self.B.shape[0]:self.pred_horizon*self.B.shape[0],(self.pred_horizon-1)*self.B.shape[0]:self.pred_horizon*self.B.shape[0]] = self.F

        self.Q_tilde = Q_tilde 
        self.G = G 
        self.M = G.T @ Q_tilde @ G + R_tilde
        

    def discretize(self,dt):
 
        self.dt = dt

        G = np.zeros([self.B.shape[0]*self.pred_horizon,self.B.shape[1]*self.pred_horizon])


        tmp_B = dt * self.B 

        del_A = np.eye(self.B.shape[0]) + dt * self.A 

        self.del_A_pow = [del_A]

        for i in range(0,self.pred_horizon):
            for j in range(0,self.pred_horizon-i):
                G[(j+i)*self.B.shape[0]:(j+i+1)*self.B.shape[0],(j)*self.B.shape[1]:(j+1)*self.B.shape[1]] = tmp_B 
            
            self.del_A_pow.append(del_A @ self.del_A_pow[-1])
            tmp_B = del_A @ tmp_B

        Q_tilde = np.zeros([self.pred_horizon*self.A.shape[0],self.pred_horizon*self.A.shape[1]])

        Q_tilde[0:(self.pred_horizon-1)*self.A.shape[0],0:(self.pred_horizon-1)*self.A.shape[1]] = np.kron(np.eye(self.pred_horizon-1),self.Q)

        Q_tilde[(self.pred_horizon-1)*self.B.shape[0]:self.pred_horizon*self.B.shape[0],(self.pred_horizon-1)*self.B.shape[0]:self.pred_horizon*self.B.shape[0]] = self.F

        R_tilde = np.kron(np.eye(self.pred_horizon),self.R)

        self.G = G 
        self.M = G.T @ Q_tilde @ G + R_tilde
        self.Q_tilde = Q_tilde 

    def setup_qp_linear(self, x0, u0, xd_tiled):
        #REMARK: here x0,u0 denotes where the control process starts.
        x0 = x0.reshape(-1,1)
        #xd = xd.reshape(-1,1)

        c = np.zeros([self.pred_horizon*self.B.shape[0],1])
        fmr_mat = np.eye(self.B.shape[0])
        for i in range(self.pred_horizon):
            f_minus_r = self.f0 - self.A @ self.x_lin - self.B @ self.u_lin
            c[i*self.B.shape[0]:(i+1)*self.B.shape[0]] = self.del_A_pow[i] @ x0 + self.dt * fmr_mat @ f_minus_r
            fmr_mat = fmr_mat + self.del_A_pow[i]

        #xd_tilde = np.tile(xd,[self.pred_horizon,1])

        self.XD = (xd_tiled - c).T @ self.Q_tilde @ self.G
        #print(c)

    def setup_qp_by_ref(self,x0,u0,xd):
        x0 = x0.reshape(-1,1)
        xd = xd.reshape(-1,1)

        c = np.zeros([self.pred_horizon*self.B.shape[0],1])
        fmr_mat = np.eye(self.B.shape[0])
        for i in range(self.pred_horizon):
            f_minus_r = self.f0 - self.A @ self.x_lin - self.B @ self.u_lin
            c[i*self.B.shape[0]:(i+1)*self.B.shape[0]] = self.del_A_pow[i] @ x0 + self.dt * fmr_mat @ f_minus_r
            fmr_mat = fmr_mat + self.del_A_pow[i]

        self.c = c
        self.XD = (xd- c).T @ self.Q_tilde @ self.G
        #print(c)

    def compute_c(self,x0):
        c = np.zeros([self.pred_horizon*self.B.shape[0],1])
        fmr_mat = np.eye(self.B.shape[0])
        for i in range(self.pred_horizon):
            f_minus_r = self.f0 - self.A @ self.x_lin - self.B @ self.u_lin
            c[i*self.B.shape[0]:(i+1)*self.B.shape[0]] = self.del_A_pow[i] @ x0 + self.dt * fmr_mat @ f_minus_r
            fmr_mat = fmr_mat + self.del_A_pow[i]

        return c

    def solve_qp(self,extra = None):
        if extra is None:
            def loss(x):
                
                return 0.5 * np.dot(x.T, np.dot(self.M, x)) - np.dot(self.XD, x)
            
            def jac(x):
                return np.dot(x.T, self.M) - self.XD
        else:
            def loss(x):
                
                return 0.5 * np.dot(x.T, np.dot(self.M, x)) - np.dot(self.XD, x) + np.dot(extra,x)
            
            def jac(x):
                return np.dot(x.T, self.M) - self.XD + extra
        

        


        opt = {'disp':False}

        u0 = np.zeros([self.pred_horizon*self.B.shape[1],1]).reshape(-1)

        if self.cons is not None:
            res_cons = optimize.minimize(loss,u0, jac=jac,
                                        method='SLSQP', options=opt,constraints=self.cons)
        else:
            res_cons = optimize.minimize(loss,u0, jac=jac,
                                        method='SLSQP', options=opt)
        
        return res_cons.x 

    def apply_control(self,x0,u0):

        return x0 + self.dt * self.f(x0,u0)
    



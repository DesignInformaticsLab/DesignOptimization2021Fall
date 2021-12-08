import numpy as np
import torch


# Helper functions
def split(array, slices):
    return np.array(np.array_split(array, slices)).squeeze()


# Problem formulation
def objective(x):
    return x[0]**2 + (x[1] - 3)**2

def df(x):
    return np.array([2*x [0], 2*(x[1]-3)])

def ineq_constraints(x):
    return np.array([x[1]**2 - 2*x[0], (x[1] - 1)**2 + 5*x[0] - 15])

def dg(x):
    return np.array([[-2, 2*x[1].item()],[5, 2*(x[1].item() - 1)]])

def lagrangian(x, mu):
    return objective(x) + np.dot(split(mu, 2),split(ineq_constraints(x), 2))

def dL(x, mu):    
    return df(x) + np.dot(mu.transpose(), dg(x)).transpose()



def QP(x, f, A, W, lam = None, mu = None,  h = None, g = None):
    '''
    QP subproblem with active set strategy
    Inputs:
    Returns: s, mu
    '''
    satisfied = False # Iterate QP Subproblem until satisfied = True
    while  not satisfied:
        h_bar = []
        active_set = []
        # Add active constraints
        for i, g_i in enumerate(g):
            if g_i == 0:
                h_bar.append(g_i)
                active_set.append(i)

        # Derivatives of all constraints 
        g_x = np.array([[-2, 2*x[1].item()],[5, 2*(x[1].item() - 1)]])
        
        # Compute A, fx for active set
        A = np.asarray(g_x[active_set])
        fx = np.asarray([[2*x[0].item(), 2*(x[1].item()-3)]]).transpose()
        
        # solve for s and lam_bar
        if len(active_set) > 0:
            soln = np.matmul(np.linalg.inv(np.asarray([[W, A.transpose()], [A, np.zeros((len(active_set), len(active_set)))]])), np.block([-fx, -h_bar]))
            s, mu_bar =  soln[0:2], soln[2:,]
            # constraint multiplier feasibility conditions
            if mu_bar.all() > 0 and (g_x[active_set]*s + h_bar).all() <= 0:
                satisfied = True
            elif mu_bar.any() <= 0:
                i_ = np.argmin(mu_bar) # Exclude index
                active_set = np.delete(active_set, i_) # update active set
            elif (g_x[active_set]*s + h_bar).any() > 0:
                j_ = np.argmax(g_x[active_set]*s + h_bar)
                active_set = np.append(active_set, j_)
        elif len(active_set) == 0:
            h_bar = np.zeros((2,1))
            # soln = np.matmul(np.linalg.inv(np.block([[W, np.zeros((2,2))], [np.zeros((2,2)), np.zeros((2,2))]])), np.stack([-fx, -h_bar]).reshape(4,1))
            soln = np.matmul(np.linalg.inv(W), -fx)
            s = soln
            mu_bar = np.zeros((2,1))
            satisfied = True
    return s, mu_bar


def line_search(f, A, s, lam = None, mu = None, h = None, g = None):
    
    return alpha


def BFGS(W, x_new, x_old):
    # Compute y_k
    y_k  = df(x_new) - df(x_old)
    # Compute theta
    if np.dot(dx.transpose(),y_k) >= 0.2*np.dot(np.dot(dx.transpose(),W),dx):
        theta = 1
    else:
        theta = (0.8*np.dot(np.dot(dx.transpose(),W),dx))/(np.dot(np.dot(dx.transpose(),W),dx) - np.dot(dx.transpose(),y_k))
    # Compute dg_k
    dg_k = theta*y_k + (1-theta)*np.dot(W,dx)
    W = W  + (np.dot(dg_k,dg_k.transpose())/np.dot(dg_k.transpose(),dx)) + (np.dot(W, dx)*np.dot(W, dx).transpose()/np.dot(np.dot(dx.transpose(),W),dx))
    return W

# Initialization
x = np.array([[1,1]]).transpose()
mu = np.zeros((2,1))
A = np.zeros((2,2))
W = np.eye(2) # 2x2 matrix as x has dim = 2
k = 0
eps = 1E-3
lam = 0

f = objective(x)
f_x = df(x)
g = ineq_constraints(x)
g_x = dg(x)
L = lagrangian(x, mu)
L_x = dL(x, mu)
solution = []
solution.append(x)
while np.linalg.norm(L_x) > eps:
    s, mu_new = QP(x, f, A, W, mu = mu, g = g)
    # alpha = line_search(f, A, s, mu = mu, g = g)
    alpha = 0.01
    dx = alpha*s
    x_new = x + dx
    # Hessian update using BFGS
    W = BFGS(W, x_new, x)
    # Update grad L
    L_x = dL(x_new, mu_new)
    x = x_new
    mu = mu_new
    solution.append(x)

import numpy as np


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
        gx = np.array([[-2, 2*x[1].item()],[5, 2*(x[1].item() - 1)]])
        
        # Compute A, fx for active set
        A = np.asarray(gx[active_set])
        fx = np.asarray([[2*x[0].item(), 2*(x[1].item()-3)]]).transpose()
        
        # solve for s and lam_bar
        if len(active_set) > 0:
            soln = np.matmul(np.linalg.inv(np.asarray([[W, A.transpose()], [A, np.zeros((len(active_set), len(active_set)))]])), np.block([-fx, -h_bar]))
            s, mu_bar =  soln[0:2], soln[2:,]
            # constraint multiplier feasibility conditions
            if mu_bar.all() > 0 and (gx[active_set]*s + h_bar).all() <= 0:
                satisfied = True
            elif mu_bar.any() <= 0:
                i_ = np.argmin(mu_bar) # Exclude index
                active_set = np.delete(active_set, i_) # update active set
            elif (gx[active_set]*s + h_bar).any() > 0:
                j_ = np.argmax(gx[active_set]*s + h_bar)
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


def BFGS()

# Initialization
x = np.asarray([[1,1]]).transpose()
f = x[0]**2 + (x[1] - 3)**2 
g = np.asarray([x[1]**2 - 2*x[0], (x[1] - 1)**2 + 5*x[0] - 15])

lam = 0
mu = np.zeros((2,1))
k = 0
eps = 1E-3
W = np.eye(2) # 2x2 matrix as x has dim = 2
L_x = np.asarray([[2*x [0], 2*(x[1]-3)]]).transpose()
A = np.zeros((2,2))

while np.linalg.norm(L_x) > eps:
    x_current = x
    s, mu = QP(x, f, A, W, mu = mu, g = g)
    # alpha = line_search(f, A, s, mu = mu, g = g)
    alpha = 0.01
    x_next = x_current + alpha*s
    W = BFGS(W, alpha*s, ,)
import numpy as np


def QP(x, lam, mu, f, A, W, h = None, g = None):
    '''
    QP subproblem with active set strategy
    '''
    # Find active set
    h_bar = []
    active_set = []
    for i, g_i in enumerate(g):
        if g_i == 0:
            h_bar.append(g_i)
            active_set.append(i)

    # Derivatives of all constraints 
    gx = np.asarray([[-2, 2*x[1]],[5, 2*(x[1] - 1)]])
    
    # Compute A, fx for active set
    A = np.asarray(gx[active_set])
    fx = np.asarray([[2*x[0], 2*(x[1]-3)]]).transpose()
    
    # solve for s and lam_bar
    s, lam_bar = np.asarray([[W, A.transpose()], [A, np.zeros(len(active_set), 2)]])
    return s, lam, mu


def line_search():
    return alpha

# Initialization
x = np.asarray([[1,1]]).tranpose()
lam = 0
mu = 0
k = 0
eps = 1E-3
W = np.eye(2) # 2x2 matrix as x has dim = 2
L_x = np.asarray([[2*x [0], 2*(x[1]-3)]]).transpose()
A = np.zeros((2,2))

while np.linalg.norm(L_x) > eps:
    s, lam, mu = QP(x, lam, mu, f, A, W, g, h)
    alpha = line_search(s, lam, mu, f, h, g, A)
    x = x + alpha*s
    W = BFGS()
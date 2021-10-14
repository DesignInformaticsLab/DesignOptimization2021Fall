# Problem 1 : Least squares fitting for vapor-liquid equilibrium
import torch
import numpy as np
import matplotlib.pyplot as plt

SMOOTH = 1e-5


def model(X, A, p_sat):
    # Question : Temporary variables and the computational graph ...
    k1 = A[0]*(A[1]*X[1]/(A[0]*X[0] + A[1]*X[1]))
    k2 = A[1]*(A[0]*X[0]/(A[0]*X[0] + A[1]*X[1]))
    t1 = X[0]*torch.exp((k1**2))*p_sat[0]
    t2 = X[1]*torch.exp((k2**2))*p_sat[1]
    return t1 + t2


# Define the variables to be optimized over
A = torch.tensor([1.9,1.6], requires_grad = True, dtype = torch.float64)
# Data
X = np.asarray([np.arange(0,1.1,0.1), 1-np.arange(0,1.1,0.1)])
X = torch.from_numpy(X)
p = torch.tensor([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.8, 17.5], requires_grad =  False, dtype = torch.float64)

# Constants
# a = torch.tensor([[8.07131, 1730.63, 233.426],[7.43155, 1554.679, 240.337]], requires_grad= False)
# T = 20
# p_sat_water = torch.tensor(a[0][0] - (a[0][1]/(T + a[0][2])))
# p_sat_dioxane = torch.tensor(a[1][0] - (a[1][1]/(T + a[1][2])))
p_sat = torch.tensor([1.2424, 1.4598], requires_grad = False, dtype = torch.float64)


def phi(alpha,p, A):
    return np.sum((p - model(X, A, p_sat)).detach().numpy()**2) - alpha*0.8*np.dot(A.grad, A.grad)

def line_search(X, p, A, p_sat):
    alpha = 0.001  # initialize step size
    while phi(alpha, p, A)<np.sum((p - model(X, A-alpha*A.grad, p_sat)).detach().numpy()**2):  # if f(x+a*d)>phi(a) then backtrack. d is the search direction
        alpha = 0.5*alpha
    return alpha
n_iter = 100
for i in range(n_iter):
    # Define the loss function
    loss = torch.sum((p - model(X, A, p_sat))**2)
    loss.backward()
    alpha = line_search(X = X, p = p, A = A, p_sat = p_sat)
    print("Alpha:", alpha, "Loss:", loss.data.numpy(), "Gradient:", A.grad.numpy(), "Parameter:", A.data.numpy())

    with torch.no_grad():
        A -= alpha * A.grad
        if np.linalg.norm(A.grad) < 0.1:
            break
        A.grad.zero_()
p_final = model(X, A, p_sat)
print("Model: ", p_final)
print("Truth: ", p)
plt.plot(p_final.detach().numpy())
plt.plot(p.detach().numpy())



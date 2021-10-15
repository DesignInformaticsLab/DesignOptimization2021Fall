# A simple example of using PyTorch for gradient descent
import torch as t
from torch.autograd import Variable

"""
# Define a variable, make sure requires_grad=True so that PyTorch can take gradient with respect to this variable
# Initial guesses
x = Variable(t.tensor([1.0, 0.0]), requires_grad=True)

# Define a loss
# REPLACE with function trying to minimize (See notes)
loss = (x[0] - 1)**2 + (x[1] - 2)**2

# Take gradient
loss.backward()

# Check the gradient. numpy() turns the variable from a PyTorch tensor to a numpy array.
# Provides gradient
x.grad.numpy()



# Let's examine the gradient at a different x.
x.data = t.tensor([2.0, 1.0])
loss = (x[0] - 1)**2 + (x[1] - 2)**2
print(type(loss))
loss.backward()
x.grad.numpy()
"""

# Equilibrium Relation (A12 and A21 are the unknowns)
# p = x1*exp(A12*(A21*x2/(A12*x1+A21*x2))**2)*pw + x2*exp(A21*(A12*x1/(A12*x1 + A21*x2))**2)*pd
# Use gradient descent to calculate A12 and A21 given the measured data

# Antoine Equation
T = 20
a1_w = 8.07131
a2_w = 1730.63
a3_w = 233.426
a1_d = 7.43155
a2_d = 1554.679
a3_d = 240.337
pw = 10**(a1_w - (a2_w/(T+a3_w)))
pd = 10**(a1_d - (a2_d/(T+a3_d)))

# Measured Data
x1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
p_measured = [28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]
x2 = []
for i in range(len(x1)):
    x2.append(round(1 - x1[i], 2))



# Here is a code for gradient descent without line search
# Initial guesses for A12 and A21, respectively
g = Variable(t.tensor([1.0, 1.0]), requires_grad=True)

# Fix the step size
a = 0.01

# Start gradient descent
# Termination Criterion: If norm of gradient is larger than a certain error, then continue...
for i in range(1000):  # TODO: change the termination criterion
    p = x1[0]*t.exp(g[0]*(g[1]*x2[0]/(g[0]*x1[0]+g[1]*x2[0]))**2)*pw + x2[0]*t.exp(g[1]*(g[0]*x1[0]/(g[0]*x1[0] + g[1]*x2[0]))**2)*pd
    loss = (p - p_measured[0])**2
    loss.backward()

    # no_grad() specifies that the operations within this context are not part of the computational graph, i.e., we don't need the gradient descent algorithm itself to be differentiable with respect to x
    with t.no_grad():
        g -= a * g.grad

        # need to clear the gradient at every step, or otherwise it will accumulate...
        g.grad.zero_()

print(g.data.numpy())
print(loss.data.numpy())

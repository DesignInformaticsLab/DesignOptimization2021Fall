"""Solve the following problem using Python SciPy.optimize.
Please attach your code and results.
Specify your initial guesses of the solution.
If you change your initial guess, do you find different solutions?
(100 points)"""

<<<<<<< HEAD
from scipy.optimize import minimize, rosen, rosen_der
=======
from scipy.optimize import minimize
>>>>>>> 5088e0127579fe1dac9dc674f218fe4bd18044e3

# The following is the function I'm trying to minimize


def fun(x): return (x[0] - x[2])**2 + \
        (x[1] + x[2] - 2)**2 + (x[3] - 1)**2 + (x[4] - 1)**2

"""
The following are the four constriants
x1 + 3*x2 = 0
x3 + x4 - 2*x5 = 0
x2 - x5 = 0
-10 <= x_i <= 10 ~i = 1, ..., 5
"""
cons = ({'type': 'eq', 'fun': lambda x: x[0] + 3*x[1]},
        {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2*x[4]},
        {'type': 'eq', 'fun': lambda x: x[1] - x[4]})
bnds = ((-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10))
res = minimize(fun, (-9, 0, 9, 2, -7), method='SLSQP',
               bounds=bnds, constraints=cons)
print(res.fun)

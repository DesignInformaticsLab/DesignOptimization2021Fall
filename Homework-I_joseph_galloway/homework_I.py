"""Solve the following problem using Python SciPy.optimize.
Please attach your code and results.
Specify your initial guesses of the solution.
If you change your initial guess, do you find different solutions?
(100 points)"""

from scipy.optimize import minimize

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

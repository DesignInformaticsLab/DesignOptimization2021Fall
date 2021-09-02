"""Solve the following problem using Python SciPy.optimize.
Please attach your code and results.
Specify your initial guesses of the solution.
If you change your initial guess, do you find different solutions?
(100 points)"""

from scipy.optimize import minimize, rosen, rosen_der

x1 = 0;
x2 = 0;
x3 = 0;
x4 = 0;
x5 = 0;

#The following is the function I'm trying to minimize
func = (x1 - x2)**2 + (x2 + x3 - 2)**2 + (x4 - 1)**2 + (x5 - 1)**2

"""The following are the four constriants
x1 + 3*x2 = 0
x3 + x4 - 2*x5 = 0
x2 - x5 = 0
-10 <= x_i <= 10 ~i = 1, ..., 5
"""

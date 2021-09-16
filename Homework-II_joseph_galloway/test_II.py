import sympy as sym
from numpy import linalg as la
import numpy as np

x, y = sym.symbols('x y')
eps = 1e-3  # Termination criterion
k = 0  # Counter

# Function comes from the following equation:
# min. (1 - 2x - 3y + 1)^2 + x^2 + (y - 1)^2
# where x is x2
# and y is x3
# x1 is solved at the end when x2 and x3 are solved
function = (1 - 2*x - 3*y + 1)**2 + x**2 + (y - 1)**2
def partial_x(x, y): return 10*x + 12*y - 8
def partial_y(x, y): return 12*x + 20*y - 14

solution_x = [5]  # Initial guess x
solution_y = [-7]  # Initial guess y
# Calculate gradient using initial guesses to calculate initial error
gradient = [partial_x(solution_x[0], solution_y[0]), partial_y(solution_x[0], solution_y[0])]

# Calculate the norm of the 2D gradient at the initial guesses
error = la.norm(gradient)


a = 0.05
while error >= eps:
    # Calculate next step using gradient descent algorithm
    solution_x.append(solution_x[k] - a*partial_x(solution_x[k], solution_y[k]))
    solution_y.append(solution_y[k] - a*partial_y(solution_x[k], solution_y[k]))

    # Update gradient norm and calculate new error using new x,y position
    gradient = [partial_x(solution_x[k], solution_y[k]), partial_y(solution_x[k], solution_y[k])]
    error = la.norm(gradient)
    k += 1  # Increase counter

# Solution for x1 now that x2 and x3 are solved for
solution_x1 = 1 - 2*solution_x[k] - 3*solution_y[k]

soln = [solution_x1, solution_x[k], solution_y[k]]
print(soln)
# print(k)
# This solution matches what I calculated in part I of this problem.

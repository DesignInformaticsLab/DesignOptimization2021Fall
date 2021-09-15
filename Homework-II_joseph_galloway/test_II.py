import sympy as sym


x, y = sym.symbols('x y')
eps = 1e-3  # Termination criterion
k = 0  # Counter

function = (1 - x)**2 + y**2
# partial_x = sym.diff(function, x)
# partial_y = sym.diff(function, y)


def partial_x(x): return 2*(1 - x)
def partial_y(y): return 2*y


initial_guess = [0, 0]  # Initial guess for x and y
solution_x = [initial_guess[0]]
solution_y = [initial_guess[1]]

a = 0.01

# while

import sympy as sym

def calc_gradient(x, y):
    pass


x, y = sym.symbols('x y')

function = (1 - x)**2 + y**2
partial_x = sym.diff(function, x)
partial_y = sym.diff(function, y)
gradient = [partial_x, partial_y]


initial_guess = [0, 1]  # Initial guess for x and y

print(gradient)

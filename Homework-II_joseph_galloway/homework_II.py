# Homework II
# Code adapted from the following sources:
# 1) https://towardsdatascience.com/hessian-matrix-and-optimization-problems-in-python-3-8-f7cd2a615371


import numpy as np
import sympy as sp
from sympy import symbols, Eq, solve


x, y, z = sp.symbols('x y z')


# Problem I
def partial(element, function):
    # The diff function is from the sympy library
    # It takes the derivative of the input function using...
    # the variable input
    # https://docs.sympy.org/latest/tutorial/calculus.html
    partial_diff = function.diff(element)
    return partial_diff


def gradient_to_zero(symbols_list, partials):
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)

    # The function "solve" solves for the unknown variable(s) in the...
    # given function(s)
    # https://problemsolvingwithpython.com/10-Symbolic-Math/10.06-Solving-Equations/
    singular = solve((partial_x, partial_y),
                     (symbols_list[0], symbols_list[1]))

    return singular


def hessian(partials_second, cross_derivatives):
    # Create the Hessian Matrix for the original function using...
    # the matrix function
    # https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
    hessianmat = np.matrix([[partials_second[0], cross_derivatives],
                            [cross_derivatives, partials_second[1]]])

    return hessianmat


def problem_I():
    print("Problem I")
    x, y = symbols('x y')  # Used for symbolic calculations
    symbols_list = [x, y]  # Used in the partial function
    # Empty arrays to hold the first partial derivative and
    # the second partial derivative
    partials, partials_second = [], []
    function = 2*x**2 - 4*x*y + 1.5*y**2 + y

    # Using x as the first element, call "partial" using x and...
    # the current function as the inputs
    # Append the result to the array partials
    for element in symbols_list:
        partial_diff = partial(element, function)
        partials.append(partial_diff)

    # Calculates the second partial derivatives of the function...
    # for the diagonal Hessian Matrix
    # Assuming the Hessian is symmetric
    cross_derivatives = partial(symbols_list[0], partials[1])

    # Repeat the process as above using the first partial derivatives currently...
    # in the partial_diff variable
    # Overwrite the partial_diff array with the second partial derivatives by ...
    # calling "partial" again.
    for i in range(0, len(symbols_list)):
        partial_diff = partial(symbols_list[i], partials[i])
        partials_second.append(partial_diff)

    # Create the Hessian Matrix for the input function
    hessianmat = hessian(partials_second, cross_derivatives)

    print("Hessian matrix of the function {0} = \n {1}".format(
        function, hessianmat))
    print("The critical point is "
          + str(gradient_to_zero(symbols_list, partials)))

    # Second half of problem I
    # Find the direction of the downslopes away from the...
    # saddle (Use Taylor's Expansion at the saddle point).

    print("____________________")


# Problem II


def problem_II():
    pass
    print("Problem II")
    print("____________________")


# Problem III


def problem_III():
    pass
    print("Problem III")
    print("____________________")


# Problem IV


def problem_IV():
    pass
    print("Problem IV")
    print("____________________")


# Problem V


def problem_V():
    pass
    print("Problem V")
    print("____________________")


def main():
    # Comment out problems not being evaluated if desired
    problem_I()
    # problem_II()
    # problem_III()
    # problem_IV()
    # problem_V()


main()

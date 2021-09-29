# Homework III
from math import exp
import matplotlib.pyplot as plt
from sympy import symbols, diff


def problem_I():
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
    p = [28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]
    x2 = []
    for i in range(len(x1)):
        x2.append(round(1 - x1[i], 2))
    # plt.plot(x1, p)
    # plt.show()

    # Initial guesses
    A12 = 0
    A21 = 0


def problem_II():
    pass


def main():
    problem_I()
    # problemII()


main()

# Homework III
from math import exp
import matplotlib.pyplot as plt
from sympy import symbols, diff


def problem_I():
    # Equilibrium Relation (A12 and A21 are the unknowns)
    # p = x1*exp(A12*(A21*x2/(A12*x1+A21*x2))**2)*pw + x2*exp(A21*(A12*x1/(A12*x1 + A21*x2))**2)*pd
    # Use gradient descent to calculate A12 and A21 given the measured data
    # A = A12
    # B = A21
    # A, B = symbols('A B', real=True)

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
    p_measured = [28.1, 34.4, 36.7, 36.9, 36.8,
                  36.7, 36.5, 35.4, 32.9, 27.7, 17.5]
    x2 = []
    for i in range(len(x1)):
        x2.append(round(1 - x1[i], 2))
    p = []
    error = []
    # plt.plot(x1, p)
    # plt.show()

    # Initial guesses
    A = [1.0]
    B = [1.0]

    # Calculate p with initial guess parameters
    for i in range(len(x1)):
        p.append(x1[i]*exp(A*(B*x2[i]/(A*x1[i]+B*x2[i]))**2)*pw
                 + x2[i]*exp(B*(A*x1[i]/(A*x1[i] + B*x2[i]))**2)*pd)
        # print(p[i])

    # Calculate each error for initial parameter guesses
    for i in range(len(x1)):
        p.append(x1[i]*exp(A*(B*x2[i]/(A*x1[i]+B*x2[i]))**2)*pw
                 + x2[i]*exp(B*(A*x1[i]/(A*x1[i] + B*x2[i]))**2)*pd)
        error.append(-1*(p_measured[i] - p[i])**2)

        # Calculate graident of function given x1[i] and x2[i]

    # plt.plot(x1, error)
    # plt.show()

    # Gradient Descent
    a = 0.05  # Learning rate
    k = 0  # Counter
    eps = 10**-3  # Acceptable error
    """while error > eps:
        # Each calculation is using x[k]
        A.append(A[k] - a*merp_A)  # Previous A - (learning rate)*partial_A
        B.append(B[k] - a*merp_B)  # Previous B - (learning rate)*partial_B

        # Calculate new gradient using updated parameters A[k+1] & B[k+1]
        # Calculate new p
        # x1[i]*exp(A*(B*x2[i]/(A*x1[i]+B*x2[i]))**2)*pw + x2[i]*exp(B*(A*x1[i]/(A*x1[i] + B*x2[i]))**2)*pd

        # Calculate new error using new p and corresponding p_measured
        # error = -1*(p_measured[i] - p[i])**2

        k += 1
        """


def problem_II():
    pass


def main():
    problem_I()
    # problemII()


main()

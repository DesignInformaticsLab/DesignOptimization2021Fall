import sympy as sym

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

# A, B, x, y = sym.symbols('A B x y')  # This cannot be used to calculate new gradient for each iteration
A, B = sym.symbols('A B')

# Set x1 and x2 equal to 1 to calculate error gradient
x1 = 1
x2 = 1
error = x1*sym.exp(A*(B*x2/(A*x1+B*x2))**2)*pw + x2 * \
                   sym.exp(B*(A*x1/(A*x1 + B*x2))**2)*pd

partial_A = error.diff(A)
partial_B = error.diff(B)

print(partial_A)
print(partial_B)


def partial_A(A, B): return 17.4732520845971*(-2*A*B**2/(A + B)**3 + B**2/(A + B)**2)*sym.exp(A*B
                                                                                              ** 2/(A + B)**2) + 28.8240995274052*(-2*A**2*B/(A + B)**3 + 2*A*B/(A + B)**2)*sym.exp(A**2*B/(A + B)**2)


def partial_B(A, B): return 17.4732520845971*(-2*A*B**2/(A + B)**3 + 2*A*B/(A + B)**2)*sym.exp(A*B
                                                                                               ** 2/(A + B)**2) + 28.8240995274052*(-2*A**2*B/(A + B)**3 + A**2/(A + B)**2)*sym.exp(A**2*B/(A + B)**2)

# Calculate derivatives given x1[i] and x2[i]
# for i in range(len(x1)):
#    error = x1[i]*sym.exp(A*(B*x2[i]/(A*x1[i]+B*x2[i]))**2)*pw + \
#                  x2[i]*sym.exp(B*(A*x1[i]/(A*x1[i] + B*x2[i]))**2)*pd
# partial_A.append(error.diff(A))
# print(partial_A)
# partial_B.append(error.diff(B))
# partial_B.append(error.diff(B))

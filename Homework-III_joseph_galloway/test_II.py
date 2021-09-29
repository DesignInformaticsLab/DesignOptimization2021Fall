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

A, B = sym.symbols('A B')
partial_A = []
partial_B = []

# Calculate derivatives given x1[i] and x2[i]
for i in range(len(x1)):
    error = x1[i]*sym.exp(A*(B*x2[i]/(A*x1[i]+B*x2[i]))**2)*pw + \
                  x2[i]*sym.exp(B*(A*x1[i]/(A*x1[i] + B*x2[i]))**2)*pd
    partial_A.append(error.diff(A))
    print(partial_A)
    partial_B.append(error.diff(B))

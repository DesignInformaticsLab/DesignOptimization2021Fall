# Homework IV
import numpy as np
import sympy as sy

#x1 = x, x2 = y, x3 = z

# Parameters
k = 0
eps = 10^-3

# Initial guess (Must satisfy equalities)
x = 1  # State variable
y = 1  # State variable
z = 1  # Decision variable

p_df_dd = np.array([2*z])
df_ds = np.array([2*x, 2*y])
dh_ds = np.array([[x/2, 2*y/5], [1, 1]])
dh_dd = np.array([[2*z/25], [-1]])

print(str(p_df_dd) + "\n")
print(str(df_ds) + "\n")
print(str(dh_ds) + "\n")
print(str(dh_dd) + "\n")

df_dd = p_df_dd - df_ds*np.linalg.inv(dh_ds) * dh_dd
norm = np.linalg.norm(df_dd)
print(norm)
"""
k = 0  # Counter
while norm**2 > eps:
    pass
    # Step 4.1 Line search
    # Step 4.2 Update decision variable
    # Step 4.3 Guess initial state variables
    # Step 4.4 Update state variables
    # Step 4.5 Calculate df_dd and update counter
    k = k + 1
"""

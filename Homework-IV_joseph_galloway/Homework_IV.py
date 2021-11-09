# Homework IV
import numpy as np
import sympy as sy


# Initial guess (Must satisfy equalities)
x = 1  # x1 State variable
y = 1  # x2 State variable
z = 1  # x3 Decision variable

p_df_dd = np.array([2*z])
df_ds = np.array([2*x, 2*y])
dh_ds = np.array([[x/2, 2*y/5], [1, 1]])
dh_dd = np.array([[2*z/25], [-1]])

"""
print(str(p_df_dd) + "\n")
print(str(df_ds) + "\n")
print(str(dh_ds) + "\n")
print(str(dh_dd) + "\n")
"""

df_dd = p_df_dd - np.matmul(np.matmul(df_ds, np.linalg.inv(dh_ds)), dh_dd)
norm = np.linalg.norm(df_dd)


# Parameters
k = 0  # Counter
eps = 10^-3  # Accepted error threshold
d = [z]  # Array to hold the state variables
s_o = []  # Array to hold the state variable initial guesses
s1 = [x]  # Array to hold the solved state variable 1
s2 = [y]  # Array to hold the solved state variable 2
alpha = 1  #Initial step size
b = 0.5  # Backtracking the alpha
t = 0.3  # Auxiliary function to terminate line search
f_alpha_d = []
f_alpha_s1 = []
f_alpha_s2 = []
phi_alpha_d = []
phi_alpha_s1 = []
phi_alpha_s2 = []



while norm**2 > eps:
    # Step 4.1 Line search
    f_alpha_d.append(d[k] - alpha*df_dd)  # Decision variable portion of f(alpha)
    f_alpha_s1.append(s1[k] + alpha*(np.transpose(np.matmul(np.matmul(np.linalg.inv(dh_ds), dh_dd), np.transpose(df_dd)))))  # State variable portion of f(alpha)
    f_alpha_s2.append(s2[k] + alpha*(np.transpose(np.matmul(np.matmul(np.linalg.inv(dh_ds), dh_dd), np.transpose(df_dd)))))  # State variable portion of f(alpha)
    print(f_alpha_d)
    print(f_alpha_s1)
    print(f_alpha_s2)

    break
"""
    phi_alpha_d = f_alpha_d[k]
    phi_alpha_s1 = f_alpha_s1[k] - alpha*t*(df_dd*np.transpose(df_dd))
    phi_alpha_s2 = f_alpha_s2[k] - alpha*t*(df_dd*np.transpose(df_dd))

    while f_alpha_d[k] > phi_alpha_d and f_alpha_s1[k] > phi_alpha_s1 and f_alpha_s2[k] > phi_alpha_s2:
        alpha = 2*b
        print(alpha)
        f_alpha_d.append(d[k] - alpha*df_dd)  # Decision variable portion of f(alpha)
        f_alpha_s1.append(s1[k] + alpha*(np.transpose(np.linalg.inv(dh_ds)*dh_dd*np.transpose(df_dd))))  # State variable portion of f(alpha)
        f_alpha_s2.append(s2[k] + alpha*(np.transpose(np.linalg.inv(dh_ds)*dh_dd*np.transpose(df_dd))))  # State variable portion of f(alpha)

        phi_alpha_d = f_alpha_d[k]
        phi_alpha_s1 = f_alpha_s1[k] - alpha*t*(df_dd*np.transpose(df_dd))
        phi_alpha_s2 = f_alpha_s2[k] - alpha*t*(df_dd*np.transpose(df_dd))


    # Step 4.2 Update decision variable (Similar to gradient descent)
    d.append(d[k] - alpha*df_dd)

    # Step 4.3 Update state variables by first calculating an initial guess of where S should be using a
    # linear approximation (New decision variable no longer satifies constraints (h=0), so state variables need to be updated)
    s_o.append(s[k] + alpha*(np.transpose(np.linalg.inv(dh_ds)*dh_dd*np.transpose(df_dd))))

    # Step 4.4 Update state variables using the initial guess (Use a solver to iteratively determine what Sk+1 should really be to make h = 0)
    #s.append(solve...)

    # Step 4.5 Calculate df_dd and update counter
    #df_dd = solve...
    k = k + 1
"""

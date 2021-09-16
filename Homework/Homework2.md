# Theory/Computation Problems

### Problem 1 (20 points) 
Show that the stationary point (zero gradient) of the function
$$
\begin{aligned}
    f=2x_{1}^{2} - 4x_1 x_2+ 1.5x^{2}_{2}+ x_2
\end{aligned}
$$
is a saddle (with indefinite Hessian). Find the directions of downslopes away from the saddle. Hint: Use Taylor's expansion at the saddle point. Find directions that reduce $f$.

### Problem 2 (50 points) 

* (10 points) Find the point in the plane $x_1+2x_2+3x_3=1$ in $\mathbb{R}^3$ that is nearest to the point $(-1,0,1)^T$. Is this a convex problem? Hint: Convert the problem into an unconstrained problem using $x_1+2x_2+3x_3=1$.

* (40 points) Implement the gradient descent and Newton's algorithm for solving the problem. Attach your codes along with a short summary including (1) the initial points tested, (2) corresponding solutions, (3) a log-linear convergence plot.

### Problem 3 (10 points) 
Let $f(x)$ and $g(x)$ be two convex functions defined on the convex set $\mathcal{X}$. 
* (5 points) Prove that $af(x)+bg(x)$ is convex for $a>0$ and $b>0$. 
* (5 points) In what conditions will $f(g(x))$ be convex?

### Problem 4 (bonus 10 points)
Show that $f({\bf x}_1) \geq f(\textbf{x}_0) + 
    \textbf{g}_{\textbf{x}_0}^T(\textbf{x}_1-\textbf{x}_0)$ for a convex function $f(\textbf{x}): \mathcal{X} \rightarrow \mathbb{R}$ and for $\textbf{x}_0$, $\textbf{x}_1 \in \mathcal{X}$. 

# Design Problems

### Problem 5 (20 points) 
Consider an illumination problem: There are $n$ lamps and $m$ mirrors fixed to the ground. The target reflection intensity level is $I_t$. The actual reflection intensity level on the $k$th mirror can be computed as $\textbf{a}_k^T \textbf{p}$, where $\textbf{a}_k$ is given by the distances between all lamps to the mirror, and $\textbf{p}:=[p_1,...,p_n]^T$ are the power output of the lamps. The objective is to keep the actual intensity levels as close to the target as possible by tuning the power output $\textbf{p}$.

* (5 points) Formulate this problem as an optimization problem. 
* (5 points) Is your problem convex?
* (5 points) If we require the overall power output of any of the $n$ lamps to be less than $p^*$, will the problem have a unique solution?
* (5 points) If we require no more than half of the lamps to be switched on, will the problem have a unique solution?

Problem 1.0.1 
$$f = 2x_{1}^{2} - 4x_1x_2 + 1.5x_2^2$$
The stationary point of the function is obtained by taking the gradient and setting it to zero: \
$ \rho = \begin{bmatrix}4x_1 - 4x_2 \\ -4x_1 + 3x_2 + 1 \end{bmatrix}$ = 0\
Solving for x_1 and x_2 gives a stationary point (1,1). \
Indefiniteness of the Hessian can be shown by computing its eigenvalues: \

$ H = \begin{bmatrix} 4 & -4 \\ -4 & -1 \end{bmatrix} $ \
$ \begin{vmatrix} 4 - \lambda & -4 \\ -4 & -1 - \lambda \end{vmatrix} = 0 $ \
$ \lambda_{1} = 10.54, \lambda_{2} = -4.54 $ \
Since the eigenvalues are both positive and negative, the Hessian is indefinite. \
To find the direction of the slopes awy from the saddle point (1,1) we need to find the points such that the difference f - f(1,1) < 0: \
$ f - f(1,1) = \frac{f_{x_1x_1}(1,1)(x_1 - 1)^2}{2} + \frac{f_{x_1x_2}(1,1)(x_1 - 1)(x_2 - 1)}{2} + \frac{f_{x_2x_2}(1,1)(x_2 - 1)^2}{2} $ \
$ f - 0.5 = 2(x_1 - 1)^2 - 2(x_1 - 1)(x_2 - 1) - 0.5(x_2 - 1)^2 < 0 $ \ 




Problem 2:
Part a: (Analytical)
The problem asks to find a point closest to (-1, 0, 1) such that it is constrained to lie on the plane $ x_1 + 2x_2 + 3x_3 = 1 $. \
The problem can be formulated as:
$ \underset{x}{\operatorname{argmin}} L(x_0, x) \\
 \text{st } x_1 + 2x_2 + 3x_3 = 1 $
\
Using Lagrange multipliers on the equality constraint: 
\
$ \underset{x}{\operatorname{argmin}} L(x, x_0) = (x_{01} - x_1)^2 + (x_{02} - x_2)^2 + (x_{03} - x_3)^2 = \lambda g(x) $ \
$ \nabla L(x) = \lambda \nabla g(x) $ \
$ g(x) = c $ \
The constraint equation works out to be: \
$ \nabla L(x) = \begin{bmatrix}2x_1 + 2 \\ 2x_2 \\ 2x_3 - 2 \end{bmatrix} = \begin{bmatrix} 0 \\ \lambda \\ 2\lambda \end{bmatrix} $ \
Together with the equation $ x_1 + 2x_2 + 3x_3 - 1 = 0 $, solving for the x's we get:\
$ x = \begin{bmatrix} -1 \\ \frac{-1}{8} \\ \frac{3}{4} \end{bmatrix} $ \





Problem 3: 
(i) Let f(x) and g(x) be two convex functions defined on the convex set X. 
The linear combination h(x) is defined as: \
$ h(x) = af(x) + bg(x) $
If f(x) and g(x) are convex then: \
$ f(\lambda x_1 + (1- \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) $ \
$ g(\lambda x_1 + (1- \lambda)x_2) \le \lambda g(x_1) + (1 - \lambda)g(x_2) $ \
$ af(\lambda x_1 + (1- \lambda)x_2) + bg(\lambda x_1 + (1- \lambda)x_2) \le a(\lambda f(x_1) + (1 - \lambda)f(x_2)) + b(\lambda g(x_1) + (1 - \lambda)g(x_2)) $ \
The right side can be rewritten as: \
$ \lambda(af(x_1) + bg(x_1)) + (1-\lambda)(af(x_2) + bg(x_2)) $ \
From the definition of h(x), the right hand side can be transformed to: \
$ \lambda h(x_1) + (1-\lambda)h(x_2) $ \
$ \square $ \



Problem 5 
(i) Formulation of optimization problem:
$ \underset{p_j}{\operatorname{min}} \sum_{k = 1}^{n} (I_k - I_t)^2 $ \
st $ 0 \le p_j \le p_{max}$ \
(ii) A formulation is convex if the objective is a convex function and the constraints have a feasible set that is convex. In  this case, the objective is convex due to the fact that $ I_k - I_t = a_k^Tp - I_t $ which is a hyperplane, and squaring a hyperplane is a quadratic function that is also convex. Moreover, the summation of k convex functions is also convex. Therefore the objective is convex. The feasible set from the constraints is convex because they form a segmented space that is a hypercube. \
(iii) If we require the power output of any of the n lamps to be less than p*, the modified optimization problem is: \
$ \underset{p_j}{\operatorname{min}} \sum_{k = 1}^{n} (I_k - I_t)^2 $ \
st $ 0 \le p_j \le p_{max}$ , \
$ \sum_{j = 1}^{k} p_j < p^*,  k = {1,...,n}$ \
The second constraint can be further split into n separate constraints. Unique solutions are possible if all the constraints are convex sets. Consider the constraint $ \sum_{j = 1}^{l} p_j \lt p^* , l \epsilon k $. The summation produces an l-dimensional half space of a hyperplane, which is a convex set. Therefore the problem has a unique solution.





```python
# PROBLEM 2 : CODE FOR GRADIENT DESCENT
print("Gradient descent solution")
import numpy as np

def objective(x):
    return (2 - (2*x[0]) - (3*x[1]))**2 + (x[0])**2 + (x[1] - 1)**2

def grad(x):
    return np.asarray([10*x[0] + 12*x[1] - 8, 12*x[0] + 20*x[1] - 14])

def phi(a,x):
    obj = objective(x)
    corr = a*np.asarray([g**2 for g in grad(x)])
    return obj - corr

def line_search(x):
    a = 1  # initialize step size
    while np.linalg.norm(phi(a,x))<objective(x-a*grad(x)):  # if f(x+a*d)>phi(a) then backtrack. d is the search direction
        a = 0.5*a
    return a

eps = 1e-3
x0 = [0,0]
k = 0
x = x0

error = np.linalg.norm(grad(x))

while error >= eps and k < 10000:  # keep searching while gradient norm is larger than eps
    a = line_search(x)
    x = x - a*grad(x)
    error = np.linalg.norm(grad(x))
    k+=1
    if k % 1000 == 0: print("Iteration:", k, "alpha:", a, "Error:", error, "x:", x)

x1 = 1 - 2*x[0] - 3*x[1]
x = [x1, x[0], x[1]]
print("Final solution:", x)

###########################################
# Problem 2: Newton's method
print("Newton's method solution: ")
import numpy as np

def objective(x):
    return (2 - (2*x[0]) - (3*x[1]))**2 + (x[0])**2 + (x[1] - 1)**2

def grad(x):
    return np.asarray([10*x[0] + 12*x[1] - 8, 12*x[0] + 20*x[1] - 14])

def hessian():
    return np.asarray([[10,12],[12,20]])

def phi(a,x):
    obj = objective(x)
    corr = a*np.asarray([g**2 for g in grad(x)])
    return obj - corr

def line_search(x):
    a = 1  # initialize step size
    while np.linalg.norm(phi(a,x))<objective(x-a*grad(x)):  # if f(x+a*d)>phi(a) then backtrack. d is the search direction
        a = 0.5*a
    return a

eps = 1e-3
x0 = [0,0]
k = 0
x = np.asarray(x0).reshape(2,1)

error = np.linalg.norm(grad(x))

while error >= eps and k < 1000:  # keep searching while gradient norm is larger than eps
    a = line_search(x)
    x = x - a*np.matmul(np.linalg.inv(hessian()), grad(x).reshape(2,1))
    error = np.linalg.norm(grad(x))
    k+=1
    if k % 100 == 0: print("Iteration:", k, "alpha:", a, "Error:", error, "x:", x)

x1 = 1 - 2*x[0] - 3*x[1]
x = [x1, x[0], x[1]]
print("Final solution:", x)


```

    Gradient descent solution
    Iteration: 1000 alpha: 0.125 Error: 0.8604186897840943 x: [-0.12581163  0.81128256]
    Iteration: 2000 alpha: 0.125 Error: 0.9204354759534935 x: [-0.12462265  0.81306602]
    Iteration: 3000 alpha: 0.125 Error: 0.9846386130992997 x: [-0.12335074  0.81497389]
    Iteration: 4000 alpha: 0.0625 Error: 0.3159960335302552 x: [-0.13659703  0.79510445]
    Iteration: 5000 alpha: 0.0625 Error: 0.3380377053348803 x: [-0.13616037  0.79575944]
    Iteration: 6000 alpha: 0.0625 Error: 0.36161684990617876 x: [-0.13569325  0.79646012]
    Iteration: 7000 alpha: 0.0625 Error: 0.38684071058438907 x: [-0.13519355  0.79720968]
    Iteration: 8000 alpha: 0.0625 Error: 0.4138240112546796 x: [-0.13465899  0.79801151]
    Iteration: 9000 alpha: 0.0625 Error: 0.442689478137391 x: [-0.13408714  0.79886928]
    Iteration: 10000 alpha: 0.0625 Error: 0.47356839797524564 x: [-0.13347541  0.79978688]
    Final solution: [-1.1324098264763962, -0.13347541131132373, 0.7997868830330145]
    Newton's method solution: 
    Final solution: [array([-1.07142857]), array([-0.14285714]), array([0.78571429])]


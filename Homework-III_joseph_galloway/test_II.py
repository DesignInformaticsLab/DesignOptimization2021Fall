from bayes_opt import BayesianOptimization


def function(x, y):
    return -((4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2)

# Bounded region of parameter space
pbounds = {'x': (-3, 3), 'y': (-2, 2)}

optimizer = BayesianOptimization(
    f=function,
    pbounds=pbounds,
    random_state=1,
)


optimizer.maximize(
    init_points=5,
    n_iter=100,
)

print(optimizer.max)

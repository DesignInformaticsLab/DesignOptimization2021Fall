# Problem 2: Bayesian Optimization
import sklearn.gaussian_process as gp
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib
matplotlib.use('tkAgg')
from matplotlib import pyplot as plt

from mpl_toolkits import mplot3d

def objective(x, mean = 0, var = 0.1):
    '''
    x : 2x1 vector of a sample in the domain of interest
    mean : scalar mean for iid noise term
    var : scalar variance for iid noise term
    '''
    noise = np.random.normal(mean, var)    
    return ((4 - (2.1*(x[0]**2)) + ((x[0]**4)/3))*(x[0]**2)) + (x[0]*x[1]) + ((-4 + 4*x[1]**2)*x[1]**2)




def expected_improvement(X_, X, model, k = 0.01):
    Y_mu, Y_sigma = model.predict(X, return_std = True)
    Y__= model.predict(X_)
    
    # Find the best value of predicted f(x) among the old samples
    with np.errstate(divide='ignore'):
        y_best = np.max(Y_sigma)
        Z = (Y_mu - y_best - k)/Y_sigma
        ei = ((Y_mu - y_best - k)*norm.cdf(Z)) + (Y_sigma*norm.pdf(Z))
        ei[Y_sigma == 0.0] = 0.0
    return X_[np.argmax(ei)]


# optimize the expected improvement
# EI function is highly non linear

x1 = np.random.uniform(-3,3,5)
x2 = np.random.uniform(-2,2,5)
X = np.asarray([[x1],[x2]]).transpose().squeeze(1)
Y = np.asarray([objective(x) for x in X])

kernel = gp.kernels.RBF()
model = gp.GaussianProcessRegressor(kernel=kernel,
                                    alpha=1e-4,
                                    n_restarts_optimizer=0,
                                    normalize_y=True)


x_bnds = np.linspace(-3,3, 50)
y_bnds = np.linspace(-2, 2, 50)
x__, y__ = np.meshgrid(x_bnds, y_bnds)
z__ = objective(x__,y__)

n_iters = 100
k = 0.01
for i in range(n_iters):
    model.fit(X,Y)
    # Select next point using expected improvement
    x1 = np.random.uniform(-3,3,1000)
    x2 = np.random.uniform(-2,2,1000)
    X_ = np.asarray([[x1],[x2]]).transpose().squeeze(1)
    x_next = expected_improvement(X_ = X_, X = X, model = model)
    y_next = objective(x_next)
    X = np.vstack((X, x_next))
    Y = np.append(Y, y_next)

x1_bnds = np.linspace(-3,3, 50)
x2_bnds = np.linspace(-2, 2, 50)
x__ = np.asarray([[x1_bnds],[x2_bnds]]).transpose().squeeze(1)
y__ = np.asarray([objective(x) for x in x__])

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(X[:,0], X[:,1], Y, color = "green")
ax.scatter3D(x1_bnds, x2_bnds, y__, color='blue')

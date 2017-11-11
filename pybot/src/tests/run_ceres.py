from ceres import optimize
import numpy as np

def func(ps):
    return np.sum(ps**2)

def grad(ps):
    return 2 * ps

x0 = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.double)

print(optimize(func, grad, x0))

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from functools import partial

class LinearRegression:
    def __init__(self, X, y, scaled=True):
        self.m = X.shape[0]
        self.n = X.shape[1]

        if scaled:
            self.x_scale = (np.mean(X, axis=0), np.std(X, axis=0))
            self.y_scale = (np.mean(X, axis=0), np.std(X, axis=0))
        else:
            self.x_scale = (0, 1)
            self.y_scale = (0, 1)
        
        self.X = self.add_ones(self.scale_inputs(X, self.x_scale))
        self.y = self.scale_inputs(y, self.y_scale)

        self.theta = np.zeros(self.n + 1)

    def scale_inputs(self, X, scale):
        return (X - scale[0]) / scale[1]
    
    def add_ones(self, X):    
        return np.hstack([np.ones((self.m, 1)), X])    
    
    def predict(self, X):
        X = self.add_ones(self.scale_inputs(X))
        y = self.hypothesis(X, self.theta)
        return y * self.y_scale[1] + self.y_scale[0]

    def hypothesis(self, X, theta):
        return np.dot(X, theta)

    def cost_function(self, theta, lamda):
        J_fit = (1 / self.m) * np.sum(0.5 * np.power(self.hypothesis(self.X, theta) - self.y, 2))
        J_reg = lamda * np.sum(np.power(theta[1:], 2)) # Omitting the constant term
        return J_fit + J_reg

    def grad(self, theta, lamda):
        grad = (1 / self.m) * np.dot(self.X.T, (self.hypothesis(self.X, theta) - self.y))
        grad[1:] = grad[1:] + (lamda / self.m) * theta[1:]

        return grad

    def fit_normal(self, lamda):
        L = np.eye(self.n + 1)
        L[0][0] = 0
        return np.linalg.pinv(np.dot(self.X.T, self.X) + lamda * L).dot(np.dot(self.X.T, self.y))

    def fit_solver(self, lamda):
        res = minimize(self.cost_function, x0=self.theta, args=(lamda,), jac=self.grad, method='BFGS') 
        return res.x
    
    def fit(self, lamda, use_solver=False):
        if use_solver:
            self.theta = self.fit_solver(lamda)
        else:
            self.theta = self.fit_normal(lamda)

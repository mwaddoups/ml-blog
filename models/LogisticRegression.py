from __future__ import division

import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    def __init__(self, X, y):
        self.m = X.shape[0]
        self.n = X.shape[1]

        self.x_scale = (np.mean(X, axis=0), np.std(X, axis=0))
        # Removed self.y_scale

        self.X = self.add_ones(self.scale_inputs(X, self.x_scale))
        # Instead of scaling y, check it's only got two values
        self.y = y
        assert np.all((y == 0) | (y == 1)), "Class values outside 0,1 in y"

        self.theta = np.zeros(self.n + 1)

    def scale_inputs(self, X, scale):
        return (X - scale[0]) / scale[1]
    
    def add_ones(self, X):    
        return np.hstack([np.ones((X.shape[0], 1)), X])    
    
    def predict(self, X):
        X = self.add_ones(self.scale_inputs(X, self.x_scale))
        y = self.hypothesis(X, self.theta)
        return y 

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def cost_function(self, theta, lamda):
        hyp = self.hypothesis(self.X, theta)
        J_fit = (-1 / self.m) * np.sum(self.y * np.log(hyp) + (1 - self.y) * np.log(1 - hyp))
        J_reg = lamda * np.sum(np.power(theta[1:], 2)) # Omitting the constant term
        return J_fit + J_reg

    def grad(self, theta, lamda):
        grad = (1 / self.m) * np.dot(self.X.T, (self.hypothesis(self.X, theta) - self.y))
        grad[1:] = grad[1:] + (lamda / self.m) * theta[1:]
        return grad

    def fit_solver(self, lamda):
        res = minimize(self.cost_function, x0=self.theta, args=(lamda,), jac=self.grad, method='BFGS') 
        return res.x
    
    def fit(self, lamda):
        self.theta = self.fit_solver(lamda)

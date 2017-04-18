from __future__ import division

import numpy as np
from scipy.optimize import minimize

class NeuralNetwork:
    # Layers = e.g. [10, 25] for 2 hidden layers, first 10, second 25
    def __init__(self, X, y, hidden_layers):
        self.m = X.shape[0]
        self.n = X.shape[1]
        self.num_layers = len(hidden_layers) + 1

        self.x_scale = (np.mean(X, axis=0), np.std(X, axis=0) + 0.00001)

        self.X = self.add_ones(self.scale_inputs(X, self.x_scale))
        self.y = y.reshape(self.m, (y.shape[0] * y.shape[1]) // self.m)
        
        self.theta = self.make_theta(hidden_layers + [self.y.shape[1]])

    def scale_inputs(self, X, scale):
        return (X - scale[0]) / scale[1]
    
    def add_ones(self, X):    
        return np.hstack([np.ones((X.shape[0], 1)), X])    
    
    def make_theta(self, hidden_layers):
        sj = self.n
        theta = []
        for num_nodes in hidden_layers:
            init = 1 / np.sqrt(sj)
            theta.append(np.random.random_sample((num_nodes, sj + 1)) * 2 * init - init)
            sj = num_nodes
        return theta
    
    def to_vec(self, theta):
        return np.concatenate([t.flatten() for t in theta])

    def from_vec(self, theta_vec):
        start_idx = 0
        theta = []
        for theta_mat in self.theta:
            target_idx = start_idx + theta_mat.shape[0] * theta_mat.shape[1]
            theta.append(theta_vec[start_idx:target_idx].reshape(theta_mat.shape))
            start_idx = target_idx
        return theta

    def predict(self, X):
        X = self.add_ones(self.scale_inputs(X, self.x_scale))
        y = self.hypothesis(X, self.theta)
        return y
        
    def classify(self, X):
        y = self.predict(X)
        if y.shape[1] <= 1:
            return (y.flatten() > 0.5).astype(int)
        else:
            return np.argmax(y, axis=1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    def hypothesis(self, X, theta):
        a = X
        for i in xrange(self.num_layers):
            z = np.dot(a, theta[i].T)
            if i != (self.num_layers - 1):
                a = self.add_ones(self.sigmoid(z))
            else:
                a = self.sigmoid(z)
        return a

    def cost_and_grad_function(self, theta_vec, lamda):
        theta = self.from_vec(theta_vec)
        
        # Forward propagate
        a = [self.X]
        z = []
        for i in xrange(self.num_layers):
            z.append(np.dot(a[i], theta[i].T))
            if i != (self.num_layers - 1):
                a.append(self.add_ones(self.sigmoid(z[i])))
            else:
                a.append(self.sigmoid(z[i]))
        
        # Cost function
        hyp = a[-1]
        J_fit = (-1 / self.m) * np.sum(self.y * np.log(hyp) + (1 - self.y) * np.log(1 - hyp))
        J_reg = (lamda / (self.m * 2)) * np.sum([np.sum(np.power(t[:,1:], 2)) for t in theta])
        J = J_fit + J_reg

        # Back propagate
        grad = [np.zeros(t.shape) for t in self.theta]
        d = a[-1] - self.y
        for i in reversed(xrange(self.num_layers)):
            grad[i] += np.dot(d.T, a[i]) / self.m
            grad[i][:, 1:] += (lamda / self.m) * theta[i][:, 1:]
            if i != 0:
                d = np.dot(d, theta[i][:,1:]) * self.sigmoid_gradient(z[i-1])
        grad = self.to_vec(grad) 
        
        return J, grad

    def fit_solver(self, lamda, verbose):
        res = minimize(self.cost_and_grad_function, x0=self.to_vec(self.theta), args=(lamda,)
                ,jac=True, method='CG', options={'disp': verbose, 'maxiter': 6000,})
        return self.from_vec(res.x)
    
    def fit(self, lamda, verbose=False):
        self.theta = self.fit_solver(lamda, verbose)

# Task 3: logistic regression written by hand with NumPy
import numpy as np

def sigmoid(z):
    # g(z) = 1 / (1 + e^-z), with z clipped to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    # h_theta(x) = g(theta^T x) for all rows at once
    return sigmoid(X @ theta)

def cost(X, y, theta):
    # J(theta) = -(1/n) * log-likelihood, with h clipped to [1e-12, 1 - 1e-12] before the log
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def fit_logistic(X, y, alpha, n_iters):
    # batch gradient descent from theta = 0, recording J(theta) after every iteration
    n = len(y)
    theta = np.zeros(X.shape[1])
    J_history = []
    for _ in range(n_iters):
        # update rule: theta_j += (alpha/n) * sum((y - h) * x_j), for every j at once
        theta = theta + (alpha / n) * (X.T @ (y - hypothesis(X, theta)))
        J_history.append(cost(X, y, theta))
    return theta, J_history

def predict(X, theta):
    # predict 1 if h_theta(x) >= 0.5, otherwise 0
    return (hypothesis(X, theta) >= 0.5).astype(int)

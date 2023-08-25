import numpy as np
import torch
import math


torch.set_default_dtype(torch.float32)


def power_iteration(A):
    A = A.detach().cpu().numpy()
    x = np.random.randn(A.shape[0])
  
    # Define the tolerance for the eigenvalue
    # and eigenvector approximations
    # (i.e. the maximum allowed difference between
    # the approximations and the actual values)
    tol = 1e-2
    
    # Define the maximum number of iterations
    max_iter = 100
    
    # Define the variable lam_prev to store the
    # previous approximation for the largest eigenvalue
    lam_prev = 0
    
    # Iteratively improve the approximations
    # for the largest eigenvalue and eigenvector
    # using the power method
    for i in range(max_iter):
        # Compute the updated approximation for the eigenvector
        x = A @ x / np.linalg.norm(A @ x)
    
        # Compute the updated approximation for the largest eigenvalue
        lam = (x.T @ A @ x) / (x.T @ x)
    
        # Check if the approximations have converged
        if np.abs(lam - lam_prev) < tol:
            break
    
        # Store the current approximation for the largest eigenvalue
        lam_prev = lam
    
    # Print the approximations for the
    # largest eigenvalue and eigenvector
    # print(float(lam))
    # print(x)
    return lam


def cosDis(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def decode(X_):
    a = []
    X = X_.numpy()
    for i in range(len(X)):
        a.append(X[i])
    return a


def relu(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = 0
    return ss


def sgn(s):
    ss = torch.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0:
            ss[i] = 0
        else:
            ss[i] = 1
    return ss


def proj(s):
    ss = s
    for i in range(len(s)):
        if s[i] < 0:
            ss[i] = (ss[i] + math.sqrt(ss[i] ** 2 + 4 * 0.001)) / 2
    return ss
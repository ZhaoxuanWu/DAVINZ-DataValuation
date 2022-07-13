import numpy as np
import torch

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

def rbf_mmd2(X, Y, sigma=0, biased=True):
    """
    Computes squared MMD using a RBF kernel.
    
    Args:
        X, Y (Tensor): datasets that MMD is computed on
        sigma (float): lengthscale of the RBF kernel
        biased (bool): whether to compute a biased mean
        
    Return:
        MMD squared
    """
    gamma = 1 / (2 * sigma**2)
    
    XX = torch.matmul(X, X.T)
    XY = torch.matmul(X, Y.T)
    YY = torch.matmul(Y, Y.T)
    
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)
    
    K_XY = torch.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = torch.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = torch.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    
    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2


################################################################################
### Linear-time MMD with Gaussian RBF kernel

# Estimator and the idea of optimizing the ratio from:
#    Gretton, Sriperumbudur, Sejdinovic, Strathmann, and Pontil.
#    Optimal kernel choice for large-scale two-sample tests. NIPS 2012.

# Caution: Might not be accurate enough

def rbf_mmd2_streaming(X, Y, sigma=0):

    n = (min(X.shape[0], Y.shape[0]) // 2) *2
    
    gamma = 1 / (2 * sigma**2)
    rbf = lambda A, B: torch.exp(-gamma * ((A - B) ** 2).sum(axis=1))
    mmd2 = (rbf(X[:n:2], X[1:n:2]) + rbf(Y[:n:2], Y[1:n:2])
          - rbf(X[:n:2], Y[1:n:2]) - rbf(X[1:n:2], Y[:n:2])).mean()
    return mmd2

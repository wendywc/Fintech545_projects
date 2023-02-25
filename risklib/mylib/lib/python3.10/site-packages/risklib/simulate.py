import numpy as np
import pandas as pd
import math

# Non-PSD fixes
def near_psd(a,epsilon=0):
    n= a.shape[0]
    invSD = None
    out = a.copy()
    #  Convert to correlation if we get a covariance
    if (np.diag(out)==1).sum() != n:
        invSD = np.diag(1/np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    #  Update eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    # adjust eigen values to non negative
    vals[vals<epsilon]=0
    T = 1/(vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    # Add back to variance
    if invSD is not None:
        invSD = np.diag(1/np.diag(invSD))
        out = invSD @ out @ invSD
    return out

# Higman
# The first projection
def pu(x):
    n = x.shape[0]
    x_pu = x.copy()
    for i in range(n):
        for j in range(n):
            if i==j:
                x_pu[i][j]=1
    return x_pu

# The second projection
def ps(x,w=None):
    n = x.shape[0]
    if w != None:
        # parameter pass in array of diags
        w_diag = np.diag(w)
    else:
        w_diag = np.diag(np.ones(n))
    x_w = np.sqrt(w_diag) @ x @ np.sqrt(w_diag)
    # Perform (A)+ on weighted A
    vals, vecs = np.linalg.eigh(x_w)
    vals[vals<1e-8]=0
    l = np.diag(vals)
    # The adjusted A
    x_pos = vecs @ l @ vecs.T
    w_inv = np.linalg.inv(np.sqrt(w_diag))
    out = w_inv @ x_pos @ w_inv
    return out

# Calculate Frobenius Norm
def fnorm(x):
    n = x.shape[0]
    result = 0
    for i in range(n):
        for j in range(n):
            result += x[i][j] ** 2
    return result

# k: max iteration
def higham(a,gamma0=np.inf,K=100,tol=1e-08):
    delta_s = [0]
    gamma = [gamma0]
    Y = [a]
    for k in range(1,K+1):
        R_k = Y[k-1] - delta_s[k-1]
        X_k = ps(R_k)
        delta_s_k = X_k - R_k
        delta_s.append(delta_s_k)
        Y_k = pu(X_k)
        Y.append(Y_k)
        gamma_k = fnorm(Y_k-a)
        gamma.append(gamma_k)
        if gamma_k -gamma[k-1] < tol:
            vals = np.linalg.eigh(Y_k)[0]
            if vals.min() >= 1e-8:
                break
            else:
                continue
    return Y[-1]

def chol_psd(a):
    n= a.shape[0]
    root = np.zeros((n,n))
    for j in range(n):
        # s is the sum of square of values for row j subtracted from A_jj
        s=0
        if j>0:
            s = root[j,:j].T @ root[j,:j]
        # diagonal element
        temp = a[j,j] - s
        if temp <= 0 and temp >= -1e-8:
            temp =0
        root[j,j] = math.sqrt(temp)
        # Set column to 0 for 0 eigen values
        if root[j,j] == 0:
            root[j+1:n,j] = 0
        else:
            ir = 1/root[j,j]
            for i in range(j+1,n):
                s = root[i,:j].T @ root[j,:j]
                root[i,j] = (a[i,j]-s)*ir
    return root

def normal_sim(a,nsim,seed,means=[],fixmethod=near_psd):
    eigval_min = np.linalg.eigh(a)[0].min()
    if eigval_min < 1e-08:
        a = fixmethod(a)
    l = chol_psd(a)
    m = l.shape[0]
    np.random.seed(seed)
    z = np.random.normal(size=(m,nsim))
    X = (l @ z).T
    # If mean is not zero add back
    if means.size != 0:
        if means.size != m:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            X[:,i] = X[:,i] + means[i]
    return X

# Helper function for pca_sim
def pca_vecs(cov):
    eigvalues, eigvector = np.linalg.eigh(cov)
    # Sort the eig values and vector
    vals = np.flip(eigvalues)
    vecs = np.flip(eigvector,axis=1)
    # Only use the positive eigen values
    posv_ind = np.where(vals >= 1e-8)[0]
    vals = vals[posv_ind]
    vecs = vecs[:,posv_ind]
    vals = np.real(vals)
    return vals,vecs

def vals_pct(vals,vecs,pct):
    # Total eigen values
    tv = vals.sum()
    for k in range(len(vals)):
        explained = vals[:k+1].sum()/tv
        if explained >= pct:
            break
    return vals[:k+1],vecs[:,:k+1]

def pca_sim(a,nsim,seed,means=[],pct=None):
    # Use the pca function above
    vals,vecs = pca_vecs(a)
    # If pct is given
    if pct != None:
        vals,vecs = vals_pct(vals,vecs,pct)
    B = vecs @ np.diag(np.sqrt(vals))
    m = vals.size
    np.random.seed(seed)
    r = np.random.normal(size=(m,nsim))
    out = (B @ r).T
    if means.size != 0:
        if means.size != out.shape[1]:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            out[:,i] = out[:,i] + means[i]
    return out


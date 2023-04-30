import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import statsmodels.api as sm

from risklib import fitted_model

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
def ps(x,W):
    # n = x.shape[0]
    # if w != None:
    #     # parameter pass in array of diags
    #     w_diag = np.diag(w)
    # else:
    #     w_diag = np.diag(np.ones(n))
    x_w = np.sqrt(W) @ x @ np.sqrt(W)
    # Perform (A)+ on weighted A
    vals, vecs = np.linalg.eigh(x_w)
    vals[vals<1e-8]=0
    l = np.diag(vals)
    # The adjusted A
    x_pos = vecs @ l @ vecs.T
    w_inv = np.linalg.inv(np.sqrt(W))
    out = w_inv @ x_pos @ w_inv
    return out

# Calculate Frobenius Norm
def fnorm(x,W):
    W05 = np.sqrt(W)
    W05 = W05 @ x @ W05
    # n = x.shape[0]
    # result = 0
    # for i in range(n):
    #     for j in range(n):
    #         result += x[i][j] ** 2
    return (W05 * W05).sum().sum()

# k: max iteration
def higham(a,W=None,gamma0=np.inf,K=100,tol=1e-9,epsilon = 1e-9):
    n= a.shape[0]
    if W is None:
        W = np.diag(np.ones(n))
    else:
        W = np.diag(W)
    invSD = None
    yk = a.copy()
    #  Convert to correlation if we get a covariance
    if (np.diag(yk)==1).sum() != n:
        invSD = np.diag(1/np.sqrt(np.diag(yk)))
        yk = invSD @ yk @ invSD
    y0 = yk.copy()
    delta_s = [0]
    gamma = [gamma0]
    Y = [y0]
    for k in range(1,K+1):
        R_k = Y[k-1] - delta_s[k-1]
        X_k = ps(R_k,W)
        delta_s_k = X_k - R_k
        delta_s.append(delta_s_k)
        Y_k = pu(X_k)
        Y.append(Y_k)
        gamma_k = fnorm(Y_k-y0,W)
        gamma.append(gamma_k)
        if abs(gamma_k -gamma[k-1]) < tol:
            vals = np.linalg.eigh(Y_k)[0]
            if vals.min() > -epsilon:
                break
            else:
                continue
    if k < K:
        print("Converged in {} interations".format(k))
    else:
        print("Convergance failed after {} iterations".format(K))

    # Add back to variance
    if invSD is not None:
        invSD = np.diag(1/np.diag(invSD))
        out = invSD @ Y[-1] @ invSD
    else:
        out = Y[-1]
    return out

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

def normal_sim(a,nsim,seed,means=np.array([]),fixmethod=near_psd):
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

# the % each value explained 
def pca_pct(vals):
    # Total eigen values
    tv = vals.sum()
    pct_dic = {}
    for k in range(len(vals)):
        pct = vals[:k+1].sum()/tv
        pct_dic[k]=pct
    pct_series = pd.Series(pct_dic)
    return pct_series

def pca_sim(a,nsim,seed,means=np.array([]),pct=None):
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

# fitmethod = {s:fitted_model.fit_general_t for s in tickers}
# remove the mean of R
def copula(r,nsim,seed,fitmethod):
    r_fitted = {}
    U =pd.DataFrame()
    for stock in r.columns:
        r_fitted[stock]= fitmethod[stock](r[stock])
        U[stock]= r_fitted[stock].u
    R = U.corr(method='pearson')
    # Simulate
    np.random.seed(seed)
    Nsim = pca_sim(R,nsim=nsim,seed=seed)
    Usim = norm.cdf(Nsim)
    # Change to dataframe
    Usim = pd.DataFrame(Usim,columns = r.columns)
    rsim = pd.DataFrame(columns = r.columns)
    for stock in r.columns:
        rsim[stock] = r_fitted[stock].inv_cdf(Usim[stock])
    return rsim,r_fitted

# Simulate ar1 from a series of returns
def ar1_sim(r,nsim,seed):
    ar1_fit = sm.tsa.arima.ARIMA(r, order=(1, 0, 0))
    con,beta,s= ar1_fit.fit().params[0],ar1_fit.fit().params[1],np.sqrt(ar1_fit.fit().params[2])
    np.random.seed(seed)
    esim = np.random.normal(0,s,nsim)
    rsim = np.zeros(nsim)
    for i in range(nsim):
        rsim[i]=con+r.iloc[-1]*beta+esim[i]
    return rsim

# Simulate ar1 from a series of returns with days ahead
def ar1_sim_ndays(r,ndays,nsim=10000,seed=10):
    ar1_fit = sm.tsa.arima.ARIMA(r, order=(1, 0, 0))
    con,beta,s= ar1_fit.fit().params[0],ar1_fit.fit().params[1],np.sqrt(ar1_fit.fit().params[2])
    np.random.seed(seed)
    rsim = np.zeros((nsim,ndays))
    for i in range(nsim):
        rsim[i,0] = con+ beta*r.iloc[-1] + s*np.random.normal()
        for j in range(1,ndays):
            rsim[i,j]=con+ beta*rsim[i,j-1] + s*np.random.normal()
    # rsim_cum = np.sum(rsim,axis=1)
    # psim = np.zeros(nsim)
    # for i in range(nsim):
    #     psim[i]=p0*exp(rsim_cum[i])
    return rsim
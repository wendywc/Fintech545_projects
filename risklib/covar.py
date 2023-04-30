import numpy as np
import pandas as pd

def weights_gen(lamda,t):
    # Initialize values
    tw = 0
    w = np.zeros(t)
    # calculate weights
    for i in range(t):
        w[i] = (1-lamda)*lamda ** (t-i-1)
        tw += w[i]
    for i in range(t):
        w[i] = w[i]/tw
    return w

# EW(corr+var)
def w_cov(df,lamda=0.97):
    n = df.shape[1]
    t = df.shape[0]
    w = weights_gen(lamda,t)
    means = np.array(df.mean())
    xhat = df.copy()
    for i in range(n):
        xhat.iloc[:,i]=xhat.iloc[:,i]-means[i]
    cov = xhat.multiply(w,axis=0).T @ xhat
    return cov.to_numpy()

# Pearson corr + var
def pcov(df):
    vars =df.var()
    std = np.sqrt(vars)
    # Get the pearson correlation matrix
    corr = np.corrcoef(df,rowvar=False)
    cov = np.diag(std) @ corr @ np.diag(std)
    return cov

# Pearson correlation and EW variance
def pcor_ewvar(df,lamda=0.97):
    # w_var is the diag of w_cov
    w_var = np.diag(w_cov(df,lamda))
    w_std = np.sqrt(w_var)
    corr = np.corrcoef(df,rowvar=False)
    cov = np.diag(w_std) @ corr @ np.diag(w_std)
    return cov

# EW corr + Var
def wcor_var(df,lamda=0.97):
    wcov = w_cov(df,lamda)
    w_var = np.diag(w_cov(df,lamda))
    w_std = np.sqrt(w_var)
    w_corr = np.diag(1/w_std) @ wcov @ np.diag(1/w_std)
    vars =df.var()
    std = np.sqrt(vars)
    cov = np.diag(std) @ w_corr @ np.diag(std)
    return cov

# If skipmiss Flase: pairwise
def missing_cov(df,skipmiss=True,func = pcov):
   n,m = df.shape
   nMiss= df.isnull().sum().sum()
   if nMiss == 0:
      return func(df)   
   if skipmiss:
      df_new = df.dropna()
      return func(df_new)
   else:
      out = np.zeros((m,m))
      for i in range(m):
         for j in range(i+1):
            df_sub = df.iloc[:,[i,j]]
            df_sub = df_sub.dropna()
            out[i,j] = func(df_sub)[0,1]
            if i != j:
               out[j,i]= out[i,j]
   return out

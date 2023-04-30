from scipy.optimize import minimize
import numpy as np
import pandas as pd
from math import sqrt

def riskbudget(w,cov,tickers):
    pSig = sqrt(w @ cov @ w.reshape(-1,1))
    CSD = w * (cov @ w.reshape(-1,1)).flatten()/pSig
    riskBudget = pd.DataFrame(CSD/pSig,index=tickers)
    return riskBudget

# minimize the sse of CSD
def sseCSD(w,cov):
    pSig = sqrt(w @ cov @ w.reshape(-1,1))
    # CSD = w # sigma * W/pSig
    CSD = w * (cov @ w.reshape(-1,1)).flatten()/pSig
    # mean of CSD
    mCSD = CSD.mean()
    dCSD = CSD -mCSD
    # sse
    sse = (dCSD * dCSD).sum()
    return 100000*sse

# Input Covar as np array
def parity_eq(tickers,covar):
    n = len(tickers)
    w0 = np.array([1/n]*n)
    w_minsse = minimize(sseCSD,w0,args = covar,
                    constraints=({'type':'ineq','fun': lambda x: x},{'type':'ineq','fun': lambda x: 1-x},{'type':'eq','fun': lambda x:x.sum()-1}))
    w = w_minsse.x
    result = pd.DataFrame(columns=['stock','w','riskbudget','sigma'])
    result['stock'] = tickers
    result['w'] = w
    result['riskbudget'] = riskbudget(w,covar,tickers).to_numpy()
    result['sigma'] = np.sqrt(np.diag(covar))
    return result

# Add non equal risk budget
def sseCSD2(w,cov,rb):
    pSig = sqrt(w @ cov @ w.reshape(-1,1))
    # CSD = w # sigma * W/pSig
    CSD = (w * (cov @ w.reshape(-1,1)).flatten()/pSig)/rb
    # mean of CSD
    mCSD = CSD.mean()
    dCSD = CSD -mCSD
    # sse
    sse = (dCSD * dCSD).sum()
    return 100000*sse

def parity_rb(tickers,covar,rb):
    n = len(tickers)
    w0 = np.array([1/n]*n)
    w_minsse = minimize(sseCSD2,w0,args=(covar,rb),
                    constraints=({'type':'ineq','fun': lambda x: x},{'type':'ineq','fun': lambda x: 1-x},{'type':'eq','fun': lambda x:x.sum()-1}))
    w = w_minsse.x
    result = pd.DataFrame(columns=['stock','w','riskbudget','rb'])
    result['stock'] = tickers
    result['w'] = w
    result['riskbudget'] = riskbudget(w,covar,tickers).to_numpy()
    result['rb'] = rb
    return result
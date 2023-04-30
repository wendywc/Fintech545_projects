import pandas as pd
import numpy as np
from math import exp,log
from sklearn.linear_model import LinearRegression

def attribution(upreturns,w0,tickers):
    """
    upreturns: the returns dataframe to calculate 
    w0: weight of last period
    return: output of return and vol attribution
    """
    # Convert returns dataframe to matrix
    returnsmat = upreturns[tickers].to_numpy()
    # Number of days
    t = returnsmat.shape[0]
    # NUmber of assets
    n = returnsmat.shape[1]
    # Return of each period t
    preturn = np.zeros(t)
    # weights of each period
    weights = np.zeros((t,n))
    # The weight of last period
    lastw = w0
    # Calculate the return for each time period
    for i in range(t):
        weights[i,:] = lastw
        lastw = lastw * (1+returnsmat[i,:])
        # Portfolio return of time t is the sum of the updated weights
        pR = lastw.sum()
        # Normalize the weights
        lastw = lastw/pR
        preturn[i]=pR-1
    # put portfolio return into upreturns dataframe
    upreturns['Portfolio']=preturn
    # Calculate total return R 
    R = exp(np.log(1+preturn).sum())-1
    # K = GR/R
    K = log(1+R)/R
    # k_t = ln(1+R_t)/KR_t
    carinok = (np.log(1+preturn)/preturn)/K
    # return attribution: A = k_t*w*r
    r_attri = carinok.reshape(1,-1) @ (returnsmat * weights) 
    # Output the return attribution and returns
    attribution = pd.DataFrame(index=['Total Return','Return attribution','Vol Attribution'],columns=tickers+['Portfolio'])
    # Calculate the total return for each stock
    attribution.loc['Total Return'] = np.exp(np.log(upreturns[tickers+['Portfolio']]+1).sum())-1
    attribution.loc['Return attribution']= np.append(r_attri,attribution.loc['Total Return','Portfolio'])
    # Vol attribution: sigma_p * Beta, w*r_i = beta * r_p
    Y = returnsmat * weights
    X = preturn.reshape(-1,1)
    # Using OLS
    model=LinearRegression()
    model.fit(X,Y)
    betas = model.coef_
    cSD = betas * preturn.std(ddof=1)
    # Add Vol attribution to output
    attribution.loc['Vol Attribution'] = np.append(cSD.T[0],preturn.std(ddof=1))
    return attribution

def factor_attribution(upffdata,upreturns,w0,betas,tickers,xnames):
    """
    upffdata: the returns dataframe of FF factors
    upreturns: the returns dataframe of stocks
    w0: weight of last period
    betas: the ff betas
    xnames: names of factors 
    Return a dataframe of return and vol attribution for each factor
    """
    # Convert dataframe to matrix
    returnsmat = upreturns[tickers].to_numpy()
    returnsff = upffdata[xnames].to_numpy()
    # Number of days
    t = returnsmat.shape[0]
    # NUmber of assets
    n = returnsmat.shape[1]
    # Return of each period t
    preturn = np.zeros(t)
    # alpha of each time period
    residreturn = np.zeros(t)
    # weights of each period
    weights = np.zeros((t,n))
    # factor weights 
    factorweights = np.zeros((t,returnsff.shape[1]))
    # The weight of last period
    lastw = w0

    # Calculate the return for each time period
    for i in range(t):
        weights[i,:] = lastw 
        # factor weight : sum w_i * beta_ij
        factorweights[i,:]= lastw @ betas
        # update weights by returns
        lastw = lastw * (1+returnsmat[i,:])
        # Portfolio return of time t is the sum of the updated weights
        pR = lastw.sum()
        # Normalize the weights
        lastw = lastw/pR
        preturn[i]=pR-1
        # residual alpha
        residreturn[i] = pR-1 - (factorweights[i,:] * returnsff[i,:]).sum()
    # put portfolio return and alpha into upffreturns dataframe
    upffdata['Alpha'] = residreturn
    upffdata['Portfolio']=preturn

    # Calculate total return R 
    R = exp(np.log(1+preturn).sum())-1
    # K = GR/R
    K = log(1+R)/R
    # k_t = ln(1+R_t)/KR_t
    carinok = (np.log(1+preturn)/preturn)/K
    # return attribution: A = k_t*w*r
    r_attri = carinok.reshape(1,-1) @ (returnsff * factorweights)
    # return attribution of alpha
    r_alpha = (carinok * residreturn).sum()
    # Output the return attribution and returns
    attribution = pd.DataFrame(index=['Total Return','Return attribution','Vol Attribution'],columns=xnames+['Alpha','Portfolio'])
    # Calculate the total return for each stock
    attribution.loc['Total Return'] = np.exp(np.log(upffdata[xnames+['Alpha','Portfolio']]+1).sum())-1
    attribution.loc['Return attribution']= np.append(r_attri,[r_alpha,attribution.loc['Total Return','Portfolio']])
    # Vol attribution: sigma_p * Beta, w*r_i = beta * r_p
    Y = np.concatenate((returnsff * factorweights,residreturn.reshape(-1,1)),axis=1)
    X = preturn.reshape(-1,1)
    # Using OLS
    model=LinearRegression()
    model.fit(X,Y)
    B = model.coef_
    cSD = B * preturn.std(ddof=1)
    # Add Vol attribution to output
    attribution.loc['Vol Attribution'] = np.append(cSD.T[0],preturn.std(ddof=1))
    return attribution
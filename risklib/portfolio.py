import numpy as np
import pandas as pd
from math import sqrt
from scipy.optimize import minimize

def return_calculate(df,method):
    # Check if datecolumn exists
    if df.columns[0]=='Date':
        ind = df.columns[1:]
        datesig = True
    else:
        ind = df.columns
        datesig = False
    p = df.loc[:,ind]
    n = p.shape[1]
    t = p.shape[0]
    p2 = np.zeros((t-1,n))
    for i in range(t-1):
        for j in range(n):
            p2[i,j]=p.iloc[i+1,j]/p.iloc[i,j]
    # Choose the method
    if method.upper()== "DISCRETE":
        p2 = p2 -1
    elif  method.upper()== "LOG":
        p2 = np.log(p2)
    else:
        raise Exception("Method be either discrete or log")
    # Add DateColumn back to data if datecolumn exists
    out = pd.DataFrame(data=p2,columns=ind)
    if datesig == True:
        out.insert(0,'Date',np.array(df.loc[1:,'Date']))
    return out

# Get stock tickers in this portfolio
def port_tickers(portname,portdata):
    # filter tickers in the portfolio
    if portname == "All":
        tickers = portdata["Stock"]
    else:
        tickers = portdata[portdata["Portfolio"]==portname]["Stock"]
    return tickers

def port_cal(tickers,stockdata,portdata):
    prices = stockdata[tickers]
    port = portdata[portdata["Stock"].isin(tickers)]
    # Get the current price and holdings
    cur_prices = prices.iloc[-1,:]
    holdings = port["Holding"].to_numpy()
    pv = (cur_prices * holdings).sum()
    return cur_prices,holdings,pv

# covar = np.diag(std)@ corr @np.diag(std)
# Calculate the optimal portfolio
def optimize_port(er,covar,rf):
    n = len(er)
    def sharpe_cal(w):
        r = w.T @ er.reshape(-1,1)
        std = sqrt(w.T @ covar @ w)
        sharpe = (r-rf)/std
        return -sharpe
    #Initial weights
    w0 = np.array([1/n]*n).reshape(-1,1)
    w_optmize = minimize(sharpe_cal,w0,
                         constraints=({'type':'ineq','fun': lambda x: x},{'type':'ineq','fun': lambda x: 1-x},{'type':'eq','fun': lambda x:x.sum()-1}))
    sharpe = -w_optmize.fun
    return w_optmize.x,sharpe


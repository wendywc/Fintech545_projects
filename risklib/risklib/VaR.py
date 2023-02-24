import numpy as np
import pandas as pd
from scipy.stats import norm
from . import covar, portfolio, riskstats, simulate


# Delta Normal
def delta_norm(port,stockdata,portdata,alpha=0.05):
    r,cur_price,cur_value,portinfo = portfolio.port_cal(port, stockdata, portdata)
    sigma = covar.w_cov(r, 0.94)
    # Number of prices underlying
    n= cur_price.size
    delta = np.zeros(n)
    # Derivative is 1 for stocks
    for i in range(n):
        delta[i] = portinfo.iloc[i,2]*cur_price[i]/cur_value
    # scaler: sqrt(delta.T * sigma * delta)
    delta = pd.DataFrame(delta,index=sigma.index)
    scaler = np.sqrt(delta.T @ sigma @ delta)
    VaR = -cur_value*norm.ppf(alpha)*scaler
    VaR_pct = -norm.ppf(alpha)*scaler
    return VaR.iloc[0,0],VaR_pct.iloc[0,0]

# Historical Simulation
def sim_his(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = portfolio.port_cal(port, stockdata, portdata)
    # Sample from historical data with replacement
    np.random.seed(seed)
    r_sim = r.sample(nsim,replace=True)
    # Calculate the new prices * return
    p_new = (1+r_sim).mul(cur_price)
    # Calulate portfolio value for each draw
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = riskstats.VaR(profit, alpha)
    es = riskstats.ES(profit, alpha)
    return var,es

# Monte Carlo
def sim_mc(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = portfolio.port_cal(port, stockdata, portdata)
    # Remove the mean for returns
    r_h = r.sub(r.mean(),axis=1)
    sigma= covar.w_cov(r_h, 0.94)
    r_sim = simulate.pca_sim(sigma, nsim, seed, means=r.mean(), pct=None)
    # Calculate the new prices
    r_sim =pd.DataFrame(r_sim,columns=r.columns)
    p_new = (r_sim+1).mul(cur_price)
    # Calulate portfolio value for each draw
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = riskstats.VaR(profit, alpha)
    es = riskstats.ES(profit, alpha)
    return var,es
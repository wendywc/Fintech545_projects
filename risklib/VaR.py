import numpy as np
import pandas as pd
from scipy.stats import norm
from . import covar, portfolio, riskstats, simulate


# Delta Normal
def delta_norm(r,cur_prices,holdings,alpha=0.05):
    sigma = covar.w_cov(r, 0.94)
    # Number of prices underlying
    n= cur_prices.size
    delta = np.zeros(n)
    cur_value = (cur_prices * holdings).sum()
    # Derivative is 1 for stocks
    for i in range(n):
        delta[i] = holdings[i]*cur_prices[i]/cur_value
    # scaler: sqrt(delta.T * sigma * delta)
    scaler = np.sqrt(delta @ sigma @ delta.reshape(-1,1))[0]
    VaR = -cur_value*norm.ppf(alpha)*scaler
    VaR_pct = -norm.ppf(alpha)*scaler
    return VaR,VaR_pct

# Historical Simulation
def sim_his(r,cur_prices,holdings,seed,nsim=10000,alpha=0.05):
    # Sample from historical data with replacement
    np.random.seed(seed)
    r_sim = r.sample(nsim,replace=True)
    # Calculate the new prices * return
    p_new = (r_sim+1).mul(cur_prices)
    # Calulate portfolio value for each draw
    cur_value = (cur_prices * holdings).sum()
    port_value = p_new.mul(holdings).sum(axis=1)
    profit = port_value- cur_value
    var = riskstats.VaR(profit, alpha)
    es = riskstats.ES(profit, alpha)
    return var,es

# Monte Carlo
# cur_prices: pandas series; holdings: numpy array
def var_mc(r,cur_prices,holdings,seed,nsim=10000,alpha=0.05):
    # Remove the mean for returns
    r_h = r-r.mean()
    sigma= covar.w_cov(r_h, 0.94)
    r_sim = simulate.pca_sim(sigma, nsim, seed,pct=None)
    # Calculate the new prices
    r_sim =pd.DataFrame(r_sim,columns=r.columns)
    p_new = (r_sim+1).mul(cur_prices)
    # Calulate portfolio value for each draw
    cur_value = (cur_prices * holdings).sum()
    port_value = p_new.mul(holdings).sum(axis=1)
    profit = port_value- cur_value
    var = riskstats.VaR(profit, alpha)
    es = riskstats.ES(profit, alpha)
    return var,es
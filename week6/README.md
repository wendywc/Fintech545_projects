Import all needed packages

```python
from math import log,sqrt,exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.optimize import fsolve
import statsmodels.api as sm
from risklib import riskstats
```

## Problem1

Set initial assumptions

```python
# Initial assumptions
underlying =165
strike = 180
days = 14
dayyear = 365
rf = 0.0425
# dividend yield
q=0.0053
```

Calculate time to maturity

```python
#ttm
ttm = days/dayyear
```

Define the funtion for generalized Black Scholes Merton

```python
# generalized Black Scholes Merton
# rf = b       -- Black Scholes 1973
# b = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
# b = 0        -- Black 1976 futures option model
# b,r = 0      -- Asay 1982 margined futures option model
# b = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency
def gbsm(underlying,strike,ttm,rf,b,ivol,call=True):
    d1 = (log(underlying/strike)+(b+ivol**2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)
    if call:
        return underlying * exp((b-rf)*ttm) * norm(0,1).cdf(d1) - strike*exp(-rf*ttm)* norm(0,1).cdf(d2)
    else:
        return strike*exp(-rf*ttm)* norm(0,1).cdf(-d2) - underlying * exp((b-rf)*ttm) * norm(0,1).cdf(-d1)
```

Calculated the value of options under a range of implied volatility

```python
# calculate for different volatility
vols = np.arange(0.1,0.8,0.01)
call_prices=[]
put_prices = []
for vol in vols:
    call_prices.append(gbsm(underlying,strike,ttm,rf,rf-q,vol,call=True))
    put_prices.append(gbsm(underlying,strike,ttm,rf,rf-q,vol,call=False))
```

Plot Call option value vs. Implied Vol

```python
plt.plot(vols,call_prices,label="Call option")
plt.xlabel("Volatility")
plt.ylabel("Value")
plt.legend()
```

Plot Put option value vs. Implied Vol

```python
plt.plot(vols,put_prices,label="Put option")
plt.xlabel("Volatility")
plt.ylabel("Value")
plt.legend()
```

## Problem2

Read in file for AAPL options 

```python
aapl_options = pd.read_csv("AAPL_Options.csv")
```

Calculated ttm based on current date

```python
# Calculate ttm
curdate= datetime(2023,3,3)
aapl_options['Expiration']=aapl_options['Expiration'].astype("datetime64[ns]")
aapl_options['ttm'] = aapl_options['Expiration'].apply(lambda x: round((x-curdate).days/365,6))
aapl_options['Type'] = aapl_options['Type'].apply(lambda x: bool(x=="Call"))
```

Define the function for solving implied volatility

```python
# Solve Implied volatility using gbsm
def implied_vol(underlying,strike,ttm,rf,b,call,value,initvol):
    def sol_vol(x,underlying,strike,ttm,rf,b,call,value):
        return gbsm(underlying,strike,ttm,rf,b,x,call=call)-value
    vol = fsolve(sol_vol,initvol,args=(underlying,strike,ttm,rf,b,call,value))
    return vol[0]
```

Set initial assumptions

```python
# Initial assumptions
underlying =151.03
rf = 0.0425
# dividend yield
q=0.0053
```

calculated the implied volatility for each option using the current price

```python
# Apply the function to each option
def row_fun(row,underlying,rf,b):
    vol = implied_vol(underlying,row['Strike'],row['ttm'],rf,b,row['Type'],row['Last Price'],0.2)
    return vol
aapl_options['Implied Vol'] = aapl_options.apply(row_fun,args=(underlying,rf,round(rf-q,4)),axis=1)
```

Separate the dataset into call and put

```python
aapl_call = aapl_options[aapl_options['Type']==True]
aapl_put = aapl_options[aapl_options['Type']==False]
```

Plot the call option implied vol vs. strike price

```python
# Plot call options
plt.plot(aapl_call['Strike'],aapl_call['Implied Vol'])
plt.ylabel("Implied Volatility")
plt.xlabel("Strike Price")
```

plot the put option implied vol vs. strike price

```python
# Plot put options
plt.plot(aapl_put['Strike'],aapl_put['Implied Vol'])
plt.ylabel("Implied Volatility")
plt.xlabel("Strike Price")
```

Put together two plots

```python
#Put together call and put
plt.plot(aapl_call['Strike'],aapl_call['Implied Vol'],label="Call")
plt.plot(aapl_put['Strike'],aapl_put['Implied Vol'],label="Put")
plt.axvline(underlying,color = "darkred",label="Current Price",linestyle="--")
plt.ylabel("Implied Volatility")
plt.xlabel("Strike Price")
plt.legend()
```

## Problem3

Read in the portfolio data

```python
port_options = pd.read_csv("Problem3.csv")
```

Calculate the ttm based on current date

```python
# Calculate ttm
curdate= datetime(2023,3,3)
port_options['ExpirationDate']=port_options['ExpirationDate'].astype("datetime64[ns]")
port_options['ttm'] = port_options['ExpirationDate'].apply(lambda x: round((x-curdate).days/365,6))
port_options['OptionType'] = port_options['OptionType'].apply(lambda x: bool(x=="Call") if pd.notna(x) else x)
```

Set initial assumptions

```python
# Initial assumptions
underlying =151.03
rf = 0.0425
# dividend yield
q=0.0053
```

Calculate implied vol using the current price given

```python
# Caculate implied vol
def row_fun_port(row,underlying,rf,b):
    vol = implied_vol(underlying,row['Strike'],row['ttm'],rf,b,row['OptionType'],row['CurrentPrice'],0.2)
    return vol
port_options['Implied Vol'] = port_options[port_options['Type']=='Option'].apply(row_fun_port,args=(underlying,rf,round(rf-q,4)),axis=1)
```

Define the function to calculate an asset given the underlying price

```python
# Calculate value for each option given the underlying
def value_cal(row,underlying,rf,b,daysahead):
    if row['Type'] == 'Stock':
        return underlying
    else:
        return gbsm(underlying,row['Strike'],row['ttm']-daysahead/365,rf,b,row['Implied Vol'],row['OptionType'])*row['Holding']
```

Set the range for underlying

```python
# Set the range for underlying
underlying_list = np.linspace(100,200,num=21).tolist()
```

Create the dataframe to store the portfolio values

```python
# Create the dataframe to store the portfolio values
port_values = pd.DataFrame(port_options['Portfolio'])
```

calculate the value of each asset given an underlying 

```python
for underlying in underlying_list:
    port_values[underlying]=port_options.apply(value_cal,args=(underlying,rf,round(rf-q,4),0),axis=1)
```

Group assets by the portfolio name

```python
# Portfolio values under the range
portfolio_range = port_values.groupby('Portfolio').sum().T
```

Plot the value of each portfolio under a range of underlying price

```python
# Plot the shape for each portfolio
fig, axs = plt.subplots(nrows=3, ncols=3,figsize=(12,8))
for i in range(len(portfolio_range.columns)):
    x_ax = i // 3
    y_ax = i % 3
    port_name = portfolio_range.columns[i]
    axs[x_ax,y_ax].plot(portfolio_range.index,portfolio_range[port_name])
    axs[x_ax,y_ax].set_title(port_name)
fig.subplots_adjust(wspace=0.3, hspace=0.4)
```

Read in the daily prices

```python
daily_prices = pd.read_csv("DailyPrices.csv")
```

Calculate the log return of AAPL and demean the series

```python
# Calculate the log return of AAPL
aapl_log = np.log(daily_prices['AAPL']/daily_prices['AAPL'].shift(1))
# Demean the series
aapl_log = aapl_log-aapl_log.mean()
```

Simulation function using AR1 given the days ahead and current price

```python
# Simulate using AR1
def ar1_sim(r,ndays,p0,nsim=10000,seed=10):
    ar1_fit = sm.tsa.arima.ARIMA(r, order=(1, 0, 0))
    con,beta,s= ar1_fit.fit().params[0],ar1_fit.fit().params[1],np.sqrt(ar1_fit.fit().params[2])
    np.random.seed(seed)
    rsim = np.zeros((nsim,ndays))
    for i in range(nsim):
        rsim[i,0] = con+ beta*r.iloc[-1] + s*np.random.normal()
        for j in range(1,ndays):
            rsim[i,j]=con+ beta*rsim[i,j-1] + s*np.random.normal()
    rsim_cum = np.sum(rsim,axis=1)
    psim = np.zeros(nsim)
    for i in range(nsim):
        psim[i]=p0*exp(rsim_cum[i])
    return psim
```

Simulate the underlying price

```python
# simulated underlying price
underlying_sim = ar1_sim(aapl_log,10,underlying,seed=20)
```

Calculate the simulated portfolio PL using the current value of the portfolio 

```python
# The current value of portfolios
port_current = port_options.apply(value_cal,args=(underlying,rf,round(rf-q,4),0),axis=1)
pl_list = []
for i in range(len(underlying_sim)):
    pl = port_options.apply(value_cal,args=(underlying_sim[i],rf,round(rf-q,4),10),axis=1) - port_current
    pl_list.append(pl)
pl_sim = pd.concat(pl_list,axis=1)
```

Transpose the simulated PL

```python
pl_sim.set_index(port_options['Portfolio'],inplace=True)
port_sim = pl_sim.groupby(level=0).sum().T
```

Calculate the mean of PL

```python
# calculate mean
port_mean = port_sim.mean(axis=0)
port_stat = pd.DataFrame(port_mean,columns=['Mean'])
```

Calculate the VaR and ES of PL

```python
# Calculate VaR and ES
vars = []
ess =[]
for col in port_sim.columns:
    vars.append(riskstats.VaR(port_sim[col].values))
    ess.append(riskstats.ES(port_sim[col].values))
port_stat['VaR'] = vars
port_stat['ES'] = ess
```

Save the results

```python
# Save the results
port_stat.to_csv('portstat.csv')
```


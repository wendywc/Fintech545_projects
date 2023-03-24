Import packages needed
```python
from math import exp,log,sqrt
from scipy.stats import norm
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from risklib.option import gbsm
from risklib.riskstats import VaR,ES
```

## Problem1

Define the function for closed form gbsm greek formulas
```python
# The closed form greek formulas
def cf_greeks(underlying,strike,ttm,rf,b,ivol,call=True):
    d1 = (log(underlying/strike)+(b+ivol**2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)
    if call:
        delta = exp((b-rf)*ttm) * norm(0,1).cdf(d1)
        theta = - underlying*exp((b-rf)*ttm)*norm(0,1).pdf(d1)*ivol/(2*sqrt(ttm)) \
        - (b-rf)*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(d1)-rf*strike*exp(-rf*ttm)*norm(0,1).cdf(d2)
        #rho for rf != b
        rho = ttm*strike*exp(-rf*ttm)*norm(0,1).cdf(d2)-ttm*underlying*norm(0,1).cdf(d1)*exp((b-rf)*ttm)
        carry_rho = ttm*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(d1)
    else:
        delta = exp((b-rf)*ttm) * (norm(0,1).cdf(d1)-1)
        theta = - underlying*exp((b-rf)*ttm)*norm(0,1).pdf(d1)*ivol/(2*sqrt(ttm)) \
        + (b-rf)*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(-d1)+ rf*strike*exp(-rf*ttm)*norm(0,1).cdf(-d2)
        rho = -ttm*strike*exp(-rf*ttm)*norm(0,1).cdf(-d2)+ttm*underlying*norm(0,1).cdf(-d1)*exp((b-rf)*ttm)
        carry_rho = -ttm*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(-d1)
    gamma = norm(0,1).pdf(d1)* exp((b-rf)*ttm)/ (underlying*ivol*sqrt(ttm))
    vega = underlying * exp((b-rf)*ttm) * norm(0,1).pdf(d1) * sqrt(ttm)
    # return greeks
    return delta,gamma,vega,theta,rho,carry_rho
```

Inputs for porblem1
```python
# Inputs
current_price = 165
strike = 165
curr_date = datetime(2022,3,13)
expiration = datetime(2022,4,15)
rf = 0.0425
q=0.0053
ivol = 0.2
ttm = (expiration-curr_date).days/365
```

print out closed form gbsm greeks for call
```python
# print out the greeks using closed form formula for call:
delta,gamma,vega,theta,rho,carry_rho = cf_greeks(current_price,strike,ttm,rf,rf-q,ivol,True)
print("Greeks for call:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("carry_rho: {}".format(carry_rho))
```

print out closed form gbsm greeks for put
```python
# print out the greeks using closed form formula for put:
delta,gamma,vega,theta,rho,carry_rho = cf_greeks(current_price,strike,ttm,rf,rf-q,ivol,False)
print("Greeks for Put:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("carry_rho: {}".format(carry_rho))
```

Define the function for finite difference method to calculate GBSM greeks
```python
# The finite difference method
def fd_greeks(d_s,d_v,d_t,d_r,d_b,current_price,strike,ttm,rf,b,ivol,call):
    f_diff = (gbsm(current_price+d_s,strike,ttm,rf,b,ivol,call=call)-gbsm(current_price,strike,ttm,rf,b,ivol,call=call))/d_s
    b_diff = (gbsm(current_price,strike,ttm,rf,b,ivol,call=call)-gbsm(current_price-d_s,strike,ttm,rf,b,ivol,call=call))/d_s
    delta = f_diff
    gamma = (f_diff-b_diff)/d_s
    vega = (gbsm(current_price,strike,ttm,rf,b,ivol+d_v,call=call)-gbsm(current_price,strike,ttm,rf,b,ivol,call=call))/d_v
    theta = -(gbsm(current_price,strike,ttm+d_t,rf,b,ivol,call=call)-gbsm(current_price,strike,ttm,rf,b,ivol,call=call))/d_t
    rho = (gbsm(current_price,strike,ttm,rf+d_r,b,ivol,call=call)-gbsm(current_price,strike,ttm,rf,b,ivol,call=call))/d_r
    carry_rho = (gbsm(current_price,strike,ttm,rf,b+d_b,ivol,call=call)-gbsm(current_price,strike,ttm,rf,b,ivol,call=call))/d_b
    return delta,gamma,vega,theta,rho,carry_rho
```

Print out finite difference greeks for call
```python
# Print out finite difference greeks for call
delta,gamma,vega,theta,rho,carry_rho=fd_greeks(0.2,0.01,0.01,0.01,0.007,current_price,strike,ttm,rf,rf-q,ivol,True)
print("Greeks for call:")
print("Delta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("carry_rho: {}".format(carry_rho))
```

Print out finite difference greeks for Put
```python
# Print out finite difference greeks for put
delta,gamma,vega,theta,rho,carry_rho=fd_greeks(0.2,0.01,0.01,0.01,0.007,current_price,strike,ttm,rf,rf-q,ivol,False)
print("Greeks for Put:")
print("Delta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("carry_rho: {}".format(carry_rho))
```

Define the function for binomial tree without dividend
```python
# binomial tree without dividend
def bt_american(underlying,strike,ttm,rf,b,ivol,N,call):
    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    if call:
        z=1
    else:
        z=-1
    # calculate the number of nodes
    def nNode(n):
        return int((n+1)*(n+2)/2)
    # Calculate the index
    def idx(i,j):
        return nNode(j-1)+i
    nNodes = nNode(N)
    optionvalues = np.zeros(nNodes)
    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            index = idx(i,j)
            price = underlying*u**i*d**(j-i)
            optionvalues[index]=max(0,z*(price-strike))
            if j<N:
               optionvalues[index] = max(optionvalues[index],df*(pu*optionvalues[idx(i+1,j+1)]+pd*optionvalues[idx(i,j+1)]))
            # print(i,j,optionvalues[index])
    return optionvalues[0]
```

Define the function for binomial tree with dividend
```python
# binomial tree with dividend
def bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call):
    # No dividends
    if len(divamts)==0 or len(divtimes)==0:
        return bt_american(underlying,strike,ttm,rf,rf,ivol,N,call)
    # First div outside grid
    if divtimes[0]>N:
        return bt_american(underlying,strike,ttm,rf,rf,ivol,N,call)
    dt = ttm/N
    u = exp(ivol*sqrt(dt))
    d = 1/u
    pu = (exp(rf*dt)-d)/(u-d)
    pd = 1.0-pu
    df = exp(-rf*dt)
    if call:
        z=1
    else:
        z=-1
    # calculate the number of nodes
    def nNode(n):
        return int((n+1)*(n+2)/2)
    # Calculate the index
    def idx(i,j):
        return nNode(j-1)+i
    nDiv = len(divtimes)
    nNodes = nNode(divtimes[0])
    optionvalues = np.zeros(nNodes)
    for j in range(divtimes[0],-1,-1):
        for i in range(j,-1,-1):
            index = idx(i,j)
            price = underlying*u**i*d**(j-i)
            if j < divtimes[0]:
                # Times before dividend, backward method
                optionvalues[index]=max(0,z*(price-strike))
                optionvalues[index] = max(optionvalues[index],df*(pu*optionvalues[idx(i+1,j+1)]+pd*optionvalues[idx(i,j+1)]))
            else:
                valnoex = bt_american_div(price-divamts[0],strike,ttm-divtimes[0]*dt,rf,divamts[1:nDiv-1],divtimes[1:nDiv-1]-divtimes[0],ivol,N-divtimes[0],call)
                valex = max(0,z*(price-strike))
                optionvalues[index] = max(valnoex,valex)
                # print("new",i,j,optionvalues[index])
    return optionvalues[0]
```

Input for binimial tree problem1
```python
# Inputs
underlying = 165
strike = 165
curr_date = datetime(2022,3,13)
expiration = datetime(2022,4,15)
div_date = datetime(2022,4,11)
rf = 0.0425
q=0.0053
ivol = 0.2
ttm = (expiration-curr_date).days/365
N=200
divtimes = int((div_date-curr_date).days/(expiration-curr_date).days *N)
div=0.88
```

Print out value using Binomial tree for call and put with and wthout dividend
```python
# Values without dividend
call_nodiv = bt_american(underlying,strike,ttm,rf,rf,ivol,N,True)
put_nodiv = bt_american(underlying,strike,ttm,rf,rf,ivol,N,False)
print("Call option without dividend:",call_nodiv)
print("Put option without dividend:",put_nodiv)
# Values with dividend
call_div = bt_american_div(underlying,strike,ttm,rf,np.array([div]),np.array([divtimes]),ivol,N,True)
put_div = bt_american_div(underlying,strike,ttm,rf,np.array([div]),np.array([divtimes]),ivol,N,False)
print("Call option with dividend:",call_div)
print("Put option with dividend:",put_div)
```

Calculate greeks using binomial tree
```python
# Calculate greeks for bt
def fd_greeks_bt(d_s,d_v,d_t,d_r,d_d,underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call):
    f_diff = (bt_american_div(underlying+d_s,strike,ttm,rf,divamts,divtimes,ivol,N,call)-bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call))/d_s
    b_diff = (bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call)-bt_american_div(underlying-d_s,strike,ttm,rf,divamts,divtimes,ivol,N,call))/d_s
    delta = f_diff
    gamma = (f_diff-b_diff)/d_s
    vega = (bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol+d_v,N,call)- bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call))/d_v
    theta = -(bt_american_div(underlying,strike,ttm+d_t,rf,divamts,divtimes,ivol,N,call)-bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call))/d_t
    rho = (bt_american_div(underlying,strike,ttm,rf+d_r,divamts,divtimes,ivol,N,call)-bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call))/d_r
    # The derivative respect to dividend
    ddiv = (bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,ivol,N,call)-bt_american_div(underlying,strike,ttm,rf,divamts-d_d,divtimes,ivol,N,call))/d_r
    return delta,gamma,vega,theta,rho,ddiv
```

Print out greeks for call without dividend
```python
# Greeks without dividend
delta,gamma,vega,theta,rho,ddiv = fd_greeks_bt(0.2,0.01,0.01,0.01,0.01,underlying,strike,ttm,rf,np.array([]),np.array([]),ivol,N,True)
print("Greeks for call:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
```

Print out greeks for put without dividend
```python
# Greeks without dividend
delta,gamma,vega,theta,rho,ddiv = fd_greeks_bt(0.2,0.01,0.01,0.01,0.01,underlying,strike,ttm,rf,np.array([]),np.array([]),ivol,N,False)
print("Greeks for Put:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
```

Print out greeks for call with dividend
```python
# Greeks with dividend
delta,gamma,vega,theta,rho,ddiv = fd_greeks_bt(0.2,0.01,0.01,0.01,0.01,underlying,strike,ttm,rf,np.array([div]),np.array([divtimes]),ivol,N,True)
print("Greeks for call:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("delta dividend : {}".format(ddiv))
```

Print out greeks for put with dividend
```python
delta,gamma,vega,theta,rho,ddiv = fd_greeks_bt(0.2,0.01,0.01,0.01,0.01,underlying,strike,ttm,rf,np.array([div]),np.array([1]),ivol,2,False)
print("Greeks for Put:")
print("Deta:{}".format(delta))
print("gamma:{}".format(gamma))
print("Vega: {}".format(vega))
print("theta: {}".format(theta))
print("rho: {}".format(rho))
print("delta dividend : {}".format(ddiv))
```

## Problem2

Read in portfolio data
```python
port_options = pd.read_csv("Problem2.csv")
daily_prices = pd.read_csv("DailyPrices.csv")
```

clean the data 
```python
port_options['ExpirationDate']=port_options['ExpirationDate'].astype("datetime64[ns]")
port_options['OptionType'] = port_options['OptionType'].apply(lambda x: bool(x=="Call") if pd.notna(x) else x)
```

Inputs for problem2
```python
# Inputs
currdate = datetime(2023,3,3)
div_date = np.array([datetime(2023,3,15)])
rf = 0.0425
# ttm = (expiration-curr_date).days/365
N=50
# divtimes = int((div_date-curr_date).days/(expiration-curr_date).days *N)
divamts = np.array([1.00])
underlying =151.03
```

Define the function to solve implied vol using binomial tree
```python
# Solve ivol using bt
def ivol_bt(underlying,strike,ttm,rf,divamts,divtimes,N,call,value,initvol):
    def sol_vol(x,underlying,strike,ttm,rf,divamts,divtimes,N,call,value):
        return bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,x,N,call)-value
    vol = fsolve(sol_vol,initvol,args=(underlying,strike,ttm,rf,divamts,divtimes,N,call,value))
    return vol[0]
```

calculate implied vol for each asset
```python
# Calculate implied vol
def row_fun_port(row,underlying,currdate,div_date,divamts,rf,N):
    ttm = ((row['ExpirationDate']-currdate).days)/365
    divtime_cal = lambda x: int((x- currdate).days/(row['ExpirationDate']-currdate).days *N)
    divtimes = np.array([divtime_cal(d) for d in div_date])
    vol = ivol_bt(underlying,row['Strike'],ttm,rf,divamts,divtimes,N,row['OptionType'],row['CurrentPrice'],0.2)
    return vol
```

```python
port_options['Implied Vol'] = port_options[port_options['Type']=='Option'].apply(row_fun_port,args=(underlying,currdate,div_date,divamts,rf,N),axis=1)
```

Define the function to calculate value for each asset using binomial tree
```python
# Calculate the value for each portfolio
def portvalue(row,underlying,currdate,div_date,divamts,rf,N,daysahead):
    # # Create a dataframe to store the values
    # port_value = pd.DataFrame(portdf['Portfolio']).copy()
    # port_value['value'] = np.NaN
    # # for each asset calculate the value using bt
    # Calculate the value using the currentdate moving ahead
    currdate += timedelta(days=daysahead)
    if row['Type']=='Stock':
            return underlying*row['Holding']
    else:
            ttm = ((row['ExpirationDate']-currdate).days)/365
            divtime_cal = lambda x: int((x- currdate).days/(row['ExpirationDate']-currdate).days *N)
            divtimes = np.array([divtime_cal(d) for d in div_date])
            value = bt_american_div(underlying,row['Strike'],ttm,rf,divamts,divtimes,row['Implied Vol'],N,row['OptionType'])
            return value * row['Holding']
    # # group assets by portfolio
    # port_result = port_value.groupby(['Portfolio']).sum()
    # return port_result
```

Get the demean return of AAPL
```python
# Calculate the log return of AAPL
aapl_log = np.log(daily_prices['AAPL']/daily_prices['AAPL'].shift(1))
# Demean the series
aapl_h = aapl_log-aapl_log.mean()
```

Simulate the current price using normal 
```python
# Simulate the price using normal
def normal_sim(r,ndays,p0,nsim=1000,seed=10):
    sigma = r.std()
    np.random.seed(seed)
    rsim = np.random.normal(0,sigma,(nsim,ndays))
    rsim_cum = np.sum(rsim,axis=1)
    psim = np.zeros(nsim)
    for i in range(nsim):
        psim[i]=p0*exp(rsim_cum[i])
    return psim
```

Calculate delta for each asset
```python
# Calculate delta for each asset
def asset_delta(row,underlying,currdate,div_date,divamts,rf,N,d):
    if row['Type']=='Stock':
        return 1
    else:
        ttm = ((row['ExpirationDate']-currdate).days)/365
        divtime_cal = lambda x: int((x- currdate).days/(row['ExpirationDate']-currdate).days *N)
        divtimes = np.array([divtime_cal(d) for d in div_date])
        delta = (bt_american_div(underlying+d,row['Strike'],ttm,rf,divamts,divtimes,row['Implied Vol'],N,row['OptionType'])- bt_american_div(underlying,row['Strike'],ttm,rf,divamts,divtimes,row['Implied Vol'],N,row['OptionType']))/d
        return delta
```

```python
port_options['delta'] = port_options.apply(asset_delta,args=(underlying,currdate,div_date,divamts,rf,N,0.2),axis=1)
```

Sum up delta and present value for each portfolio
```python
# Delta*Holding
port_options['delta_h'] = port_options['delta']*port_options['Holding']
port_options['pv'] = port_options['CurrentPrice']*port_options['Holding']
```

Simulate 10 days ahead for current price
```python
# Simulate the current prices
price_sim = normal_sim(aapl_h,10,151.03,nsim=1000,seed=10)
```

Calculated the PL from the simulated price
```python
# Calculate the PL for each simulation
# The current value of portfolios
port_curr = port_options.apply(portvalue,args=(151.03,currdate,div_date,divamts,rf,N,0),axis=1)
pl_list = []
for i in range(len(price_sim)):
    pl = port_options.apply(portvalue,args=(price_sim[i],currdate,div_date,divamts,rf,N,10),axis=1) - port_curr
    pl_list.append(pl)
pl_sim = pd.concat(pl_list,axis=1)
pl_sim.set_index(port_options['Portfolio'],inplace=True)
portpl_sim = pl_sim.groupby(level=0).sum().T
```

Calculate the mean of PL
```python
# Calculate the mean
port_mean = portpl_sim.mean(axis=0)
stat_mean = pd.DataFrame(port_mean,columns=['Mean'])
```


```python
print(port_mean)
```

Calculate VaR and ES using delta normal method 
```python
#calculate VaR and ES
def delta_var_es(row):
    r_sim = np.random.normal(0,row['scaler'],10000)
    var = VaR(r_sim)*row['pv']
    es = ES(r_sim)*row['pv']
    return var,es
```

Apply delta normal on the protfolio
```python
sigma = aapl_h.var()
port_r_cal = port_options[['Portfolio','pv','delta_h']].groupby('Portfolio').sum()
port_r_cal['dr'] = underlying/port_r_cal['pv']*port_r_cal['delta_h']
port_r_cal['scaler']= np.sqrt(port_r_cal['dr']*sigma*port_r_cal['dr'])
port_r_cal['VaR'] = port_r_cal.apply(lambda x : delta_var_es(x)[0],axis=1)
port_r_cal['ES'] = port_r_cal.apply(lambda x : delta_var_es(x)[1],axis=1)
```

Summarize the metrics
```python
# Mean ES and Var
stat = pd.concat([stat_mean,port_r_cal[['VaR','ES']]],axis=1)
```


```python
stat.to_csv('portstat.csv')
```

## Problem3

Read in the Fama data
```python
# Read in data
F_momentum_data = pd.read_csv("F-F_Momentum_Factor_daily.csv")
F_3factor_data = pd.read_csv("F-F_Research_Data_Factors_daily.csv")
```

Clean the fama data
```python
# Clean and prepare the data
F_momentum_data['Date']=F_momentum_data['Date'].astype("str")
F_3factor_data['Date']=F_3factor_data['Date'].astype("str")
F_momentum_data['Date']=F_momentum_data['Date'].apply(lambda x: datetime.strptime(x,"%Y%m%d"))
F_3factor_data['Date']=F_3factor_data['Date'].apply(lambda x: datetime.strptime(x,"%Y%m%d"))
```

```python
F_momentum_data.set_index('Date',inplace=True)
F_3factor_data.set_index('Date',inplace=True)
```


```python
F_4factor_data=pd.concat([F_momentum_data,F_3factor_data],axis=1)
# Convert to decimal not percentage
F_4factor_data = F_4factor_data/100
```


```python
print(F_4factor_data)
```

Read in the stock prices data
```python
dailyprices = pd.read_csv("DailyPrices.csv")
```

clean the data
```python
dailyprices['Date']=dailyprices['Date'].astype("datetime64[ns]")
```

The tickers for the 20 stocks used
```python
tickers = ['AAPL','META','UNH','MA',
           'MSFT','NVDA','HD','PFE',
           'AMZN','BRK-B','PG','XOM',
           'TSLA','JPM','V','DIS',
           'GOOGL','JNJ','BAC','CSCO']
```

Calculate the returns for the 20 stocks
```python
stock_r =[]
for t in tickers:
    r = np.log(dailyprices[t]/dailyprices[t].shift(1))
    stock_r.append(r)
stocks = pd.concat(stock_r,axis=1)
stocks.set_index(dailyprices['Date'],inplace=True)
stocks.dropna(inplace=True)
```

Choose the data of 1year to fit the FF model
```python
# Get the 1y data
F_4factor_1y = F_4factor_data.loc['2022-02-15':'2023-01-31']
stocks_1y = stocks.loc['2022-02-15':'2023-01-31']
```

Concat the data of each factor
```python
# The X array for model
r_mkt = F_4factor_1y['Mkt-RF'].values.reshape(-1,1)
smb = F_4factor_1y['SMB'].values.reshape(-1,1)
hml = F_4factor_1y['HML'].values.reshape(-1,1)
umd = F_4factor_1y['Mom   '].values.reshape(-1,1)
X = np.concatenate((r_mkt,smb,hml,umd),axis=1)
```

Fit the model using linear regression
```python
# Fit the model using Linear regression
model=LinearRegression()
para_dict = {}
for t in tickers:
    # r_s-r_f
    r_s= (stocks_1y[t]- F_4factor_1y['RF']).values.reshape(-1,1)
    model.fit(X,r_s)
    para_dict[t]=(model.coef_[0],model.intercept_[0])
```

Get the 10 year Fama data
```python
# 10Y facotr returns
F_4factor_10y = F_4factor_data['2013-01-31':'2023-01-31']
```


```python
print(F_4factor_10y)
```

Calculate the expected return for each stock using the fitted model and annulize the return 
```python
# er for stocks
er = []
for t in tickers:
    s_r =para_dict[t][1]+para_dict[t][0][0]*F_4factor_10y['Mkt-RF']+ para_dict[t][0][1]*F_4factor_10y['SMB'] \
    + para_dict[t][0][2]*F_4factor_10y['HML']+para_dict[t][0][3]*F_4factor_10y['Mom   ']+F_4factor_10y['RF']
    s_r = s_r.mean()*250
    er.append(s_r)
er = np.array(er)
```


```python
er
```


```python
ER_sum = pd.DataFrame(er,index=tickers)
```

Calculate the covaraince for the stocks
```python
# corr and std for stocks
corr = stocks.corr().to_numpy()
std = stocks.std().to_numpy()*sqrt(250)
covar = np.diag(std)@ corr @np.diag(std)
```

Define the function to get the super portfolio
```python
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
```

Apply the function on our portfolio
```python
weights,sharpe = optimize_port(er,covar,0.0425)
weight_res = pd.DataFrame(weights,index=tickers,columns=['Weight'])
weight_res['Weight'] = weight_res['Weight'].apply(lambda x: round(x,4))
```

Print out the ahrpe ratio of super portfolio
```python
print("Maximum Sharpe ratio:{}".format(sharpe))
```

Save the results to CSV
```python
weight_res.to_csv('weights.csv')
```


```python
ER_sum.to_csv('ER.csv')
```


```python

```

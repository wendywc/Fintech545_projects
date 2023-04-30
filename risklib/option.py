from math import exp,log,sqrt
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

# generalized Black Scholes Merton
# rf = b       -- Black Scholes 1973
# b = rf - q   -- Merton 1973 stock model where q is the continous dividend yield
# b = 0        -- Black 1976 futures option model
# b,r = 0      -- Asay 1982 margined futures option model
# b = rf - rff -- Garman and Kohlhagen 1983 currency option model where rff is the risk free rate of the foreign currency
def gbsm(underlying,strike,ttm,rf,b,ivol,call=True,greeks=True):
    # Initialize greeks
    delta,gamma,vega,theta,rho,cRho = (0,0,0,0,0,0)
    d1 = (log(underlying/strike)+(b+ivol**2/2)*ttm)/(ivol*sqrt(ttm))
    d2 = d1 - ivol*sqrt(ttm)
    if call:
        value = underlying * exp((b-rf)*ttm) * norm(0,1).cdf(d1) - strike*exp(-rf*ttm)* norm(0,1).cdf(d2)
    else:
        value = strike*exp(-rf*ttm)* norm(0,1).cdf(-d2) - underlying * exp((b-rf)*ttm) * norm(0,1).cdf(-d1)
    if greeks:
        if call:
            delta = exp((b-rf)*ttm) * norm(0,1).cdf(d1)
            theta = - underlying*exp((b-rf)*ttm)*norm(0,1).pdf(d1)*ivol/(2*sqrt(ttm)) \
            - (b-rf)*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(d1)-rf*strike*exp(-rf*ttm)*norm(0,1).cdf(d2)
            rho = ttm*strike*exp(-rf*ttm)*norm(0,1).cdf(d2)
            cRho = ttm*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(d1)
        else:
            delta = exp((b-rf)*ttm) * (norm(0,1).cdf(d1)-1)
            theta = - underlying*exp((b-rf)*ttm)*norm(0,1).pdf(d1)*ivol/(2*sqrt(ttm)) \
            + (b-rf)*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(-d1)+ rf*strike*exp(-rf*ttm)*norm(0,1).cdf(-d2)
            rho = -ttm*strike*exp(-rf*ttm)*norm(0,1).cdf(-d2)
            cRho = -ttm*underlying*exp((b-rf)*ttm)*norm(0,1).cdf(-d1)
        gamma = norm(0,1).pdf(d1)* exp((b-rf)*ttm)/ (underlying*ivol*sqrt(ttm))
        vega = underlying * exp((b-rf)*ttm) * norm(0,1).pdf(d1) * sqrt(ttm)
    return {"value":value,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho,"cRho":cRho}
    
# Solve Implied volatility using gbsm
def implied_vol(underlying,strike,ttm,rf,b,call,value,initvol):
    def sol_vol(x,underlying,strike,ttm,rf,b,call,value):
        return gbsm(underlying,strike,ttm,rf,b,x,call=call,greeks=False)["value"]-value
    vol = fsolve(sol_vol,initvol,args=(underlying,strike,ttm,rf,b,call,value))
    return vol[0]

# Binomial tree
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

# binomial tree with dividend
# divtimes = int((div_date-curr_date).days/(expiration-curr_date).days *N)
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

def ivol_bt(underlying,strike,ttm,rf,divamts,divtimes,N,call,value,initvol):
    def sol_vol(x,underlying,strike,ttm,rf,divamts,divtimes,N,call,value):
        return bt_american_div(underlying,strike,ttm,rf,divamts,divtimes,x,N,call)-value
    vol = fsolve(sol_vol,initvol,args=(underlying,strike,ttm,rf,divamts,divtimes,N,call,value))
    return vol[0]

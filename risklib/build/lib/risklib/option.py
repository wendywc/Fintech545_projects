from math import exp,log,sqrt
from scipy.stats import norm

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
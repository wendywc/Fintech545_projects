import numpy as np
import scipy.integrate as integrate

def VaR(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    return -v

# def VaR(d,alpha=0.05):
#     return -d.ppf(alpha)

def ES(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    es = a[a<=v].mean()
    return -es

# def ES(d,alpha=0.05):
#     rv=d()
#     v = VaR(d,alpha=alpha)
#     st = d.ppf(1e-12)
#     return -integrate.quad(lambda x: x*d.pdf(x),st,-v)[0]/alpha
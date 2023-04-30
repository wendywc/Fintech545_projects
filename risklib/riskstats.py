import numpy as np
import scipy.integrate as integrate

def VaR(a,alpha=0.05):
    a = np.sort(a)
    v= np.quantile(a,alpha)
    return -v

def VaR_dis(d,alpha=0.05):
    return -d.ppf(alpha)

# def VaR_df(a,alpha=0.05):
#     a = a.apply(lambda x: x.sort_values().values)
#     v= a.quantile(q=alpha,axis=0,numeric_only=True)
#     return -v

def ES(a,alpha=0.05):
    a = np.sort(a)
    v= np.quantile(a,alpha)
    es = a[a<=v].mean()
    return -es

def ES_dis(d,alpha=0.05):
    v = VaR_dis(d,alpha=alpha)
    st = d.ppf(1e-12)
    return -integrate.quad(lambda x: x*d.pdf(x),st,-v)[0]/alpha

# def ES_df(a,alpha=0.05):
#     a = a.apply(lambda x: x.sort_values().values)
#     v= a.quantile(q=alpha,axis=0,numeric_only=True)
#     es = a[a<=v].mean()
#     return -es
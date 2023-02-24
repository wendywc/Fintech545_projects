from scipy.stats import norm,t,kurtosis
import math
from scipy.optimize import minimize

class FittedModel:
    def __init__(self,errorModel,errors,u):
        self.errorModel = errorModel
        self.errors = errors
        self.u = u
    def inv_cdf(self,usim):
        sim_val = self.errorModel.ppf(usim)
        return sim_val

def fit_norm(x):
    #Mean and Std values
    m = x.mean()
    s = x.std()
    #create the error model
    errorModel = norm(m,s)
    #calculate the errors and U
    errors = x - m
    u = errorModel.cdf(x)
    return FittedModel(errorModel,errors,u)

def fit_general_t(x):
    def t_fit(vals,r):
        nu = vals[0]
        miu = vals[1]
        s = vals[2]
        ll = t.logpdf(r,df=nu,loc=miu,scale=s).sum()
        return -ll
    ll_t_res = minimize(t_fit,[2,0,x.std()],args=x,
    constraints=({'type':'ineq','fun': lambda vals: vals[0]-2},{'type':'ineq','fun': lambda vals: vals[2]}))
    nu,miu,s = ll_t_res.x[0],ll_t_res.x[1],ll_t_res.x[2]
    #create the error model
    errorModel = t(df=nu,loc=miu,scale=s)
    #calculate the errors and U
    errors = x - miu
    u = errorModel.cdf(x)
    return FittedModel(errorModel,errors,u)


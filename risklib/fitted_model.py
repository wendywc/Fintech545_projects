from scipy.stats import norm,t,kurtosis
from scipy.optimize import minimize
from math import sqrt
import numpy as np

class FittedModel:
    def __init__(self,beta,errorModel,errors,u,parameter):
        self.beta = beta
        self.errorModel = errorModel
        self.errors = errors
        self.u = u
        self.parameter = parameter
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
    parameter = {"miu":m,"sigma":s}
    return FittedModel(None,errorModel,errors,u,parameter)

#x:series not df
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
    parameter = {"df":nu,"miu":miu,"sigma":s}
    return FittedModel(None,errorModel,errors,u,parameter)

def fit_regression_t(Y,x):
    X = np.column_stack((np.ones(len(x)),x))
    def ll_t(vals,X,Y):
        # First extract s, nu,beta from the parameter passed in
        # vals: miu,s,nu,b 
        miu = vals[0]
        s = vals[1]
        nu = vals[2]
        b = vals[3:]
        xm = Y - X @ b
        # use the logpdf built-in function of t
        ll = t.logpdf(xm,nu,miu,s).sum()
        return -ll
    start_b = (np.linalg.inv(X.T @ X) @ X.T) @ Y
    e = Y - X @ start_b
    start_nu = 6.0/kurtosis(e) + 4
    start_s = sqrt(e.var()*(start_nu-2)/start_nu)
    start_miu = e.mean()
    ll_t_res = minimize(ll_t,[start_miu,start_s,start_nu,start_b[0],start_b[1]],args=(X,Y),
    constraints=({'type':'ineq','fun': lambda vals: vals[2]-2},{'type':'ineq','fun': lambda vals: vals[1]},{'type':'eq','fun': lambda vals: vals[0]}))
    miu,s,nu = ll_t_res.x[0],ll_t_res.x[1],ll_t_res.x[2]
    #create the error model
    errorModel = t(df=nu,loc=miu,scale=s)
    #Calculate the regression errors and their U values
    errors = Y - X @ ll_t_res.x[3:]
    u = errorModel.cdf(errors)
    parameter = {"df":nu,"miu":miu,"sigma":s}
    return FittedModel(ll_t_res.x[3:],errorModel,errors,u,parameter)




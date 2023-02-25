from risklib import covar
import pandas as pd

df = pd.read_csv("DailyReturn.csv").iloc[:,1:]
def test_w_cov():
    assert covar.w_cov(df,0.97).iloc[0,1] == 0.0001069456617838944

def test_pcov():
    assert (covar.pcov(df)[0,1]- 9.177633701280437e-05) < 1e-8

def test_pcor_ewvar():
    assert (covar.pcor_ewvar(df,0.97)[0,10] - 6.098180076155331e-05) < 1e-8

def test_wcor_var():
    assert (covar.wcor_var(df,0.97).iloc[0,1] - 0.00010109419236045109) < 1e-8



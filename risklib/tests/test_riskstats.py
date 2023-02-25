from risklib import riskstats

def test_VaR():
    a = [1,2,3,4,5]
    assert riskstats.VaR(a)== -1.2

def tes_ES():
    a = np.array([1,2,3,4,5])
    assert riskstats.ES(a)== -1.0
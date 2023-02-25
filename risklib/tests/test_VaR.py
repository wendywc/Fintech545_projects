from risklib import VaR
import pandas as pd

price_data = pd.read_csv("DailyPrices.csv")
port_data = pd.read_csv("portfolio.csv")

def test_delta_norm():
    assert (VaR.delta_norm("A",price_data,port_data)[0] - 5670.2029) < 1e-4
    assert (VaR.delta_norm("B",price_data,port_data)[0] - 4494.5984) < 1e-4
    assert (VaR.delta_norm("C",price_data,port_data)[0] - 3786.5890) < 1e-4

def test_sim_his():
    assert (VaR.sim_his("A",price_data,port_data,10)[0]- 9005.0672) < 1e-4
    assert (VaR.sim_his("B",price_data,port_data,10)[0]- 7001.1175) < 1e-4
    assert (VaR.sim_his("C",price_data,port_data,10)[0]- 5558.7244) < 1e-4

def test_sim_mc():
    assert (VaR.sim_mc("A",price_data,port_data,10)[0] - 5606.9019) < 1e-4
    assert (VaR.sim_mc("B",price_data,port_data,10)[0] - 4384.8329) < 1e-4
    assert (VaR.sim_mc("C",price_data,port_data,10)[0] - 3775.0590) < 1e-4
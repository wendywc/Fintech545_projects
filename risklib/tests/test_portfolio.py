from risklib import portfolio
import pandas as pd

price_data = pd.read_csv("DailyPrices.csv")
port_data = pd.read_csv("portfolio.csv")

def test_port_cal():
    assert (portfolio.port_cal("A",price_data,port_data)[0].iloc[0,1]- 0.053291) < 1e-4
    assert (portfolio.port_cal("B",price_data,port_data)[0].iloc[0,1]- 0.007987) < 1e-4
    assert (portfolio.port_cal("C",price_data,port_data)[0].iloc[0,1]- 0.008319) < 1e-4
    assert (portfolio.port_cal("A",price_data,port_data)[2] -299950.059073) < 1e-4
    assert (portfolio.port_cal("B",price_data,port_data)[2] -294385.590817) < 1e-4
    assert (portfolio.port_cal("C",price_data,port_data)[2] -270042.830527) < 1e-4
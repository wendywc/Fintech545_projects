import numpy as np
import pandas as pd

def return_calculate(df,method):
    # Check if datecolumn exists
    if df.columns[0]=='Date':
        ind = df.columns[1:]
        datesig = True
    else:
        ind = df.columns
        datesig = False
    p = df.loc[:,ind]
    n = p.shape[1]
    t = p.shape[0]
    p2 = np.zeros((t-1,n))
    for i in range(t-1):
        for j in range(n):
            p2[i,j]=p.iloc[i+1,j]/p.iloc[i,j]
    # Choose the method
    if method.upper()== "DISCRETE":
        p2 = p2 -1
    elif  method.upper()== "LOG":
        p2 = np.log(p2)
    else:
        raise Exception("Method be either discrete or log")
    # Add DateColumn back to data if datecolumn exists
    out = pd.DataFrame(data=p2,columns=ind)
    if datesig == True:
        out.insert(0,'Date',np.array(df.loc[1:,'Date']))
    return out

# Get the historical prices and holdings of a portfolio
def port_cal(port,stockdata,portdata,method="discrete"):
    if port == "All":
        port_prices = stockdata.loc[:,portdata['Stock']]
        port_info = portdata
    else:
        port_info = portdata[portdata['Portfolio']==port]
        port_prices = stockdata.loc[:,port_info['Stock']]

    cur_price = port_prices.iloc[-1,:]
    cur_value = (cur_price * np.array(port_info['Holding'])).sum()

    r = return_calculate(port_prices,method=method)
    return r,cur_price,cur_value,port_info






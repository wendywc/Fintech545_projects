import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Plot ACF and PACF of a time series
def plot_acf(ts):
    ts_acf = sm.tsa.stattools.acf(ts)
    ts_pacf = sm.tsa.stattools.pacf(ts)
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].bar(np.arange(1,11),ts_acf[1:11])
    ax[0].set_title('ACF')
    ax[1].bar(np.arange(1,11),ts_pacf[1:11])
    ax[1].set_title('PACF')
    plt.show()

# Plot Var and ES
def plot_var_es(r,var,es,method):
    sns.displot(r,kde=True)
    plt.axvline(-var, color="Red",label="VaR")
    plt.axvline(-es, color="Blue",label="ES")
    plt.title(method)
    plt.legend()
    plt.plot()

# Plot Var
def plot_var(r,var,method):
    # KDE curve
    d = sns.kdeplot(r)
    # Get x,y axis data of the plot
    x=d.lines[0].get_xydata()[:,0]
    y=d.lines[0].get_xydata()[:,1]
    # Shade area below VaR
    d.fill_between(x,y,where= (x<-var),color="red")
    plt.title(method)
    plt.show()
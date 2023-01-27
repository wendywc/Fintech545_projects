Import all the packages needed. I also upload the venv with packagaes installed, which can be activated 

```python
import numpy as np
import pandas as pd
import math
from scipy.stats import norm,skew,kurtosis,t,ttest_1samp
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
```

## Problem 1

Fist set the sample size and repeated times for t tests.

```python
# Set sample size and repeated times
sample_size = 100
samples = 1000
```
Generate the random variates from standard normal distribution, and each time calculate the skewness and kurtosis

```python
# Calculate kurtosis and skewness for samples
d = norm(0,1)
kurts =[]
skews= []
for i in range(samples):
    kurts.append(kurtosis(d.rvs(sample_size)))
    skews.append(skew(d.rvs(sample_size)))
```
Calculate the sample mean for skewness and kurtosis

```python
# Get sample mean, sample deviation for skewness and kurtosis
k_hat = np.mean(kurts)
k_std = math.sqrt(np.var(kurts,ddof =1))
s_hat = np.mean(skews)
s_std = math.sqrt(np.var(skews,ddof =1))
```
Calculate the T Statistic and P value for skewness and kurtosis. I used t.cdf function when calculating p value. And then print the result out.

```python
# Perform manual t Test
# t test for kurtosis
t_stat_k = k_hat/(k_std/math.sqrt(samples))
p_k = 2*(1-t.cdf(abs(t_stat_k),df = samples-1))
print("t stat for kurtosis : {}".format(t_stat_k))
print("p value for kurtosis : {}".format(p_k))
# t test for skewness
t_stat_s = s_hat/(s_std/math.sqrt(samples))
p_s = 2*(1-t.cdf(abs(t_stat_s),df = samples-1))
print("t stat for skewness : {}".format(t_stat_s))
print("p value for skewness : {}".format(p_s))
```
I also used included T test in the scipy package to perform T test. The results should be  the same. 

```python
# Perform included t Test
kurt_result = ttest_1samp(kurts,popmean = 0, alternative ='two-sided')
print("t stat for kurtosis : {}".format(kurt_result[0]))
print("p value for kurtosis : {}".format(kurt_result[1]))
skew_result = ttest_1samp(skews,popmean = 0, alternative ='two-sided')
print("t stat for skewness : {}".format(skew_result[0]))
print("p value for skewness : {}".format(skew_result[1]))
```

## Problem2

### OLS

First read in the data from problem2 csv file, make X and Y into vector.

```python
# Read in data from csv
data = pd.read_csv("problem2.csv")
# create vector for X and Y
X = data.iloc[:,0].to_numpy()
Y = data.iloc[:,1].to_numpy()
```
I first used OLS formula and calculate beta and error through matrix multiplication. 
x is a n*2 matrix with 1 added in front of X. 

```python
# Calculate OLS by hand
# Add 1 before the original x vector
x = np.column_stack((np.ones(len(X)),X))
x_t = np.transpose(x)
beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_t,x)),x_t),Y)
e = Y- np.matmul(x,beta)
```
Then, use the OLS model in the statsmodels package. The result should be close, but not exactly the same. We can print out the OLS summary.

```python
# Fit included OLS
model = sm.OLS(Y,X).fit()
# print(model.summary())
error = Y - model.predict(X)
print(model.summary())
```

Calculate the skewness and kurtosis of error to see if it is normally distributed.

```python
# Calculate the skewness and kurtosis of error vector
print("Skewness of error: {}".format(skew(e)))
print("kurtosis of error: {}".format(kurtosis(e)))
# Compare with the included OLS
print("Skewness of error: {}".format(skew(error)))
print("kurtosis of error: {}".format(kurtosis(error)))
```
Print out the beta estomators for OLS

```python
print("Beta for OLS",beta)
```
Plot the distribution plot for error. We can see that there are more values at tail than normal.

```python
# Plot the distribution of error
sns.displot(e, kde=True)
```

### MLE: Normal

Define the function to return ll for normal assumption. I return the absolute value of ll because I use minimize funtion below.

```python
# ll for normal
def mle_norm(s_b_array):
    # First extract s and beta from the parameter passed in
    s = s_b_array[0]
    b = np.transpose(s_b_array[1:])
    n = len(Y)
    xm = Y - np.matmul(x,b)
    s2 = s*s
    ll = -n/2 * math.log(s2*2*math.pi)-np.matmul(np.transpose(xm),xm)/(2*s2)
    # return abs of ll because I used minimize below
    return -ll
```

Use the minimize function in the scipy.optimize package. The parameters are given initilized values. Then print out the result.

```python
#Optimization
ll_normal = minimize(mle_norm,[1,1,2])
# Result
print("Normal betas:",ll_normal.x[1:])
print("Normal s:{}".format(ll_normal.x[0]))
ll_result = -mle_norm(ll_normal.x)
print("Normal ll:{}".format(ll_result))
aic_normal = 2*(1+1+1)- 2*ll_result
print("Normal AIC:{}".format(aic_normal))
bic_normal = 3*math.log(len(Y))-2*ll_result
print("Normal BIC:{}".format(bic_normal))
```

### MLE: T distribution

Similarly, define the ll for T distribution assumption. I use logpdf function in t to calculate the log of pdf, and then sum them up. 

```python
def ll_t(s_n_b_array):
    # First extract s, nu,beta from the parameter passed in
    s = s_n_b_array[0]
    nu = s_n_b_array[1]
    b = s_n_b_array[2:]
    xm = Y - x @ b
    # use the logpdf built-in function of t
    ll = t.logpdf(xm,nu,0,s).sum()
    return -ll
```
Print out the result for T. COmpare it with normal assumption. 

```python
#Optimization
ll_t_res = minimize(ll_t,[1,3,1,2])
# Result
print("T betas:",ll_t_res.x[2:])
print("T s:{}".format(ll_t_res.x[0]))
print("T df:{}".format(ll_t_res.x[1]))
ll_t_max = -ll_t(ll_t_res.x)
print("T ll:{}".format(ll_t_max))
aic_t = 2*4- 2*ll_t_max
print("T AIC:{}".format(aic_t))
bic_t = 4*math.log(len(Y))-2*ll_t_max
print("T BIC:{}".format(bic_t))
```

## Problem 3

Set parameters for MA process. The time series length is 1000, with first 50 as burn in period.

```python
#Set parameters: mu=1, theta = 0.5, sigma = 0.1 , e ~ N(0,0.1)
mu=1
theta =0.5
sigma = 0.1
n = 1000
burn_in =50
```
Repeatedly generate the y_t using the error term generated from normal distribution. 

```python
# MA(1)
#y_t = 1+ e_t + 0.5*e_t-1
y_ma1 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(1,n+burn_in):
    y_t = mu+e[i]+ theta * e[i-1]
    if i >= burn_in:
        y_ma1[i-burn_in] =y_t
```

Plot out the MA process.

```python
# Plot MA(1)
plt.plot(y_ma1)
```

Use the ARIMA model in Python to fit the simulated data. We can compare the estimators.  

```python
# Fit the simulated data using python ARIMA model
ma1_fit = sm.tsa.arima.ARIMA(y_ma1, order=(0, 0, 1))
res = ma1_fit.fit()
print(res.summary())
```
Plot the ACF plot for MA1

```python
# Show Plot of ACF
acf_ma1 = sm.tsa.stattools.acf(y_ma1)
plt.plot(np.arange(1,11),acf_ma1[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```

Plot the PACF plot for MA(1)

```python
# Show Plot of PACF
pacf_ma1 = sm.tsa.stattools.pacf(y_ma1)
plt.plot(np.arange(1,11),pacf_ma1[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```

Similarly, repeat the process for MA(2) and MA(3)

```python
# MA(2)
theta2 = 0.25
#y_t = 1+ e_t + 0.5*e_t-1 + 0.25* e_t-2
y_ma2 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(2,n+burn_in):
    y_t = mu+e[i]+ theta * e[i-1] + theta2 * e[i-2]
    if i >= burn_in:
        y_ma2[i-burn_in] =y_t
```

```python
# Fit the simulated data using python ARIMA model
ma2_fit = sm.tsa.arima.ARIMA(y_ma2, order=(0, 0, 2))
res = ma2_fit.fit()
print(res.summary())
```

```python
# Show Plot of ACF
acf_ma2 = sm.tsa.stattools.acf(y_ma2)
plt.plot(np.arange(1,11),acf_ma2[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```
    
```python
# Show Plot of PACF
pacf_ma2 = sm.tsa.stattools.pacf(y_ma2)
plt.plot(np.arange(1,11),pacf_ma2[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```

```python
# MA(3)
theta3 = 0.2
#y_t = 1+ e_t + 0.5*e_t-1 + 0.25* e_t-2 + 0.2 * e_t-3
y_ma3 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(3,n+burn_in):
    y_t = mu+e[i]+ theta * e[i-1] + theta2 * e[i-2] + theta3 * e[i-3]
    if i >= burn_in:
        y_ma3[i-burn_in] =y_t
```


```python
# Fit the simulated data using python ARIMA model
ma3_fit = sm.tsa.arima.ARIMA(y_ma3, order=(0, 0, 3))
res = ma3_fit.fit()
print(res.summary())
```

```python
# Show Plot of ACF
acf_ma3 = sm.tsa.stattools.acf(y_ma3)
plt.plot(np.arange(1,11),acf_ma3[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```

```python
# Show Plot of PACF
pacf_ma3 = sm.tsa.stattools.pacf(y_ma3)
plt.plot(np.arange(1,11),pacf_ma3[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```

Set the parameters for AR process

```python
#Set parameters: mu=1,sigma = 0.1 , e ~ N(0,0.1)
mu=1
b1 =0.5
sigma = 0.1
n = 1000
burn_in =50
```

Repeatedly generate y_t from y_t-k and normally distributed error. Use yt_last to store the time lags in the formula. 

```python
# AR(1)
#y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
yt_last = 1
y_ar1 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(1,n+burn_in):
    y_t = mu + b1 * yt_last + e[i]
    yt_last = y_t
    if i >= burn_in:
        y_ar1[i-burn_in] =y_t
```
Print out the fitting result using built in model.

```python
# Fit the simulated data using python ARIMA model
ar1_fit = sm.tsa.arima.ARIMA(y_ar1, order=(1, 0, 0))
res = ar1_fit.fit()
print(res.summary())
```
Plot ACF for AR(1)

```python
# Show Plot of ACF
acf_ar1 = sm.tsa.stattools.acf(y_ar1)
plt.plot(np.arange(1,11),acf_ar1[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```
Plot PACF for AR(1)

```python
# Show Plot of PACF
pacf_ar1 = sm.tsa.stattools.pacf(y_ar1)
plt.plot(np.arange(1,11),pacf_ar1[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```

Reapeat the process for AR(2) and AR(3)

```python
# AR(2)
b2=0.25
#y_t = 1.0 + 0.5*y_t-1 + 0.25*y_t-2 + e, e ~ N(0,0.1)
yt_last = [1,1]
y_ar2 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(1,n+burn_in):
    y_t = mu + b1 * yt_last[0] + b2 * yt_last[1] +e[i]
    yt_last[1] = yt_last[0]
    yt_last[0] = y_t
    if i >= burn_in:
        y_ar2[i-burn_in] =y_t
```

```python
# Fit the simulated data using python ARIMA model
ar2_fit = sm.tsa.arima.ARIMA(y_ar2, order=(2, 0, 0))
res = ar2_fit.fit()
print(res.summary())
```

```python
# Show Plot of ACF
acf_ar2 = sm.tsa.stattools.acf(y_ar2)
plt.plot(np.arange(1,11),acf_ar2[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```

```python
# Show Plot of PACF
pacf_ar2 = sm.tsa.stattools.pacf(y_ar2)
plt.plot(np.arange(1,11),pacf_ar2[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```

```python
# AR(3)
b3=0.1
#y_t = 1.0 + 0.5*y_t-1 + 0.25*y_t-2 + 0.1*y_t-3+ e, e ~ N(0,0.1)
yt_last = [1,1,1]
y_ar3 = np.zeros(n)
e=np.random.normal(0.0,sigma,n+burn_in)
for i in range(1,n+burn_in):
    y_t = mu + b1 * yt_last[0] + b2 * yt_last[1] + b3 * yt_last[2]+e[i]
    yt_last[2] = yt_last[1]
    yt_last[1] = yt_last[0]
    yt_last[0] = y_t
    if i >= burn_in:
        y_ar3[i-burn_in] =y_t
```

```python
# Fit the simulated data using python ARIMA model
ar3_fit = sm.tsa.arima.ARIMA(y_ar3, order=(3, 0, 0))
res = ar3_fit.fit()
print(res.summary())
```

```python
# Show Plot of ACF
acf_ar3 = sm.tsa.stattools.acf(y_ar3)
plt.plot(np.arange(1,11),acf_ar3[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("ACF")
plt.show()
```

```python
# Show Plot of PACF
pacf_ar3 = sm.tsa.stattools.pacf(y_ar3)
plt.plot(np.arange(1,11),pacf_ar3[1:11])
plt.xticks(np.arange(1,11))
plt.xlabel("t")
plt.ylabel("PACF")
plt.show()
```



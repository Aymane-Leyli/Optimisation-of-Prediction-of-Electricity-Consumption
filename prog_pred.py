import pandas as pd
import statsmodels as sm
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from datetime import datetime,date
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.seasonal
from sklearn.preprocessing import MinMaxScaler



#test de stationaritee
def stationarity_test(timeseries):
    from statsmodels.tsa.stattools import adfuller
    print("Result of Dickey-Fuller Test:")
    df_test=adfuller(timeseries,autolag="AIC")
    df_output=pd.Series(df_test[0:4],index= ["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])

    print(df_output)






donnee =pd.read_table("C:\\Users\\walid\\Desktop\\train.csv",sep = ',',header = 0)

#extraire 2 colonnes
df=donnee[['datetime','electricity_consumption']][0:552]


data1= pd.Series(df['electricity_consumption'].values,index=pd.date_range('2013-07-01 00:00:00',periods=552,freq='h'))


#examiner la stationarite de la serie:
stationarity_test(data1)


#decomposition de la serie:
decomposition = statsmodels.tsa.seasonal.seasonal_decompose(data1, model='additive')

decomposition.plot()

plt.show()





training=data1[:505]


#auto correlation
#pour determiner l ordre de MA
plot_acf(data1,lags=20)
plt.show()

#auto correlation partielle
#pour detereminer l ordre de AR
plot_pacf(data1,lags=20)
plt.show()



#ARIMA(p d q)

model= ARIMA(training, order=(2,0,1))
result=model.fit()



fcast=result.predict(start='2013-07-22 00:00:00',end='2013-07-23 23:00:00')


fcast.plot(color='b')





data1[482:].plot(color='g')
plt.show()





#residu



residu=result.resid

from scipy.stats import norm

plt.hist(residu,bins='auto',density=True,rwidth=0.85,label='Residuals')#histogramme du residu.



moy , var =norm.fit(residu)

xmin , xmax = plt.xlim()

x = np.linspace(xmin,xmax,100)

p = norm.pdf(x,moy,var)

plt.plot(x,p,'m',linewidth=2)


plt.show()



















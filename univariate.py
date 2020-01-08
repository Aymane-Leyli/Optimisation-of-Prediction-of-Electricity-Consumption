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

decomposition = statsmodels.tsa.seasonal.seasonal_decompose(data1, model='additive')

decomposition.plot()
plt.show()


stationarity_test(data1)
training=data1[:505]










# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))




import warnings



p_values = [1, 2]
d_values = range(0, 1)
q_values = range(0, 16)

warnings.filterwarnings("ignore")
evaluate_models(data1.values, p_values, d_values, q_values)







#model= ARIMA(training, order=(5,0,3))
#result=model.fit()



#fcast=result.predict(start='2013-07-22 00:00:00',end='2013-07-23 23:00:00')


#fcast.plot(color='b')





#data1[504:].plot(color='g')
#plt.show()




























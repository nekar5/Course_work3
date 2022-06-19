import math
import datetime
import numpy as np

import pandas as pd
import pandas_datareader as web
from pandas import Series, DataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style

from sqlite3 import Date

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#відрізок часу, за який беремо дані акцій
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2022, 6, 15)
#end = datetime.datetime.now()

#"тікер" компанії
ticker = "NFLX"
df = web.DataReader(ticker, 'yahoo', start, end)

#print(df)

#скорректована ціна
close_px = df['Adj Close']
#ковзаюча середня
mavg = close_px.rolling(window=100).mean()

#налаштування matplotlib + вивід на графік
mpl.rc('figure', figsize=(8, 7))
mpl.__version__
style.use('ggplot')
close_px.plot(label=ticker)
mavg.plot(label='mavg')
plt.legend()
plt.title(ticker+" stocks price graph")
plt.show()


dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0 #high/low процентаж
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0 #зміна процентажу
print(dfreg.head())

"""
#отримуємо датафрейм виду:
            Adj Close       Volume    HL_PCT  PCT_change
Date
2015-01-02  24.678253  212818400.0  3.740971   -1.849356
2015-01-05  23.983023  257142000.0  3.049410   -1.883831
2015-01-06  23.985285  263188400.0  2.635049   -0.262811
...
"""

#відкидаємо відсутні дані
dfreg.fillna(value=-99999, inplace=True)

#розмірність датафрейму
print(dfreg.shape)

#відділяємо певну частку, яку будемо прогнозувати
forecast_out = int(math.ceil(0.02 * len(dfreg))) #+-2%
print("FORECAST OUT")
print(forecast_out) #кількість днів

#відділяємо підпис (label) Adj Close, яку потім будемо прогнозувати
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

#скейлимо Х, щоб кожен (моделі) мав однаковий розподіл для лінійної регресії
X = preprocessing.scale(X)

#визначаємо вибірку для генерування (+тренування/навчання) моделей й обчислення ними даних
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#відділяємо й ідентифікуємо як у
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#вивід розмірностей
print('Dimension of X', X.shape)
print('Dimension of y', y.shape)

#розділяємо
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('train length', len(X_train)) #тренувальна (навчальна) розмірність
print('test length', len(X_test)) #тестова (експериментальна) розмірність

#лінійна регресія
clfreg = LinearRegression(n_jobs=-2)
clfreg.fit(X_train, y_train)

#поліноміальна регресія (2 степінь)
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

#поліноміальна регресія (3 степінь)
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#регресія ЛАССО (з cross validation)
clfLasso = LassoCV(eps=0.002, n_alphas=100, fit_intercept=True)
clfLasso.fit(X_train, y_train)

#оцінки (впевненості) моделей
scores = []
modelNames = ["LinearRegression", "Polynomial2", "Polynomial3", "LassoCV"]
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confLasso = clfLasso.score(X_test, y_test)
scores.append(confidencereg)
scores.append(confidencepoly2)
scores.append(confidencepoly3)
scores.append(confLasso)

#виводмио результати
print("Linear regression confidence: ", confidencereg)
print("Polynomial regression 2 confidence: ", confidencepoly2)
print("Polynomial regression 3 confidence: ", confidencepoly3)
print("LassoCV confidence: ", confLasso)
#визначаємо найкращу модель
bsIndex = [i for i, j in enumerate(scores) if j == max(scores)]
print("BEST SCORE: " + str(max(scores)) + " WITH MODEL: " + str(modelNames[bsIndex[0]]))




#вивід порівняння справжніх цін і прогнозу найкращої моделі
if (modelNames[bsIndex[0]] == "LinearRegression"):
    forecast = clfreg.predict(X_lately)
if (modelNames[bsIndex[0]] == "Polynomial2"):
    forecast = clfpoly2.predict(X_lately)
if (modelNames[bsIndex[0]] == "Polynomial3"):
    forecast = clfpoly3.predict(X_lately)
if (modelNames[bsIndex[0]] == "LassoCV"):
    forecast = clfLasso.predict(X_lately)

dfreg['Forecast'] = np.nan

#створюємо трішки зсунутим, для порівняння біч о біч (використовувалось при перевірці коду)
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

#вивід на графік
plt.plot(dfreg['Adj Close'].tail(forecast_out*4), label="Adj Close")
plt.plot(dfreg['Forecast'].shift(-forecast_out).tail(1000), label="Forecast")
plt.legend(loc=4)
plt.title("Price prediction with " + str(modelNames[bsIndex[0]]) + " Model")
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




#вивід порівняння справжніх цін і прогнозів моделей
for model in modelNames:
    if (model == "LinearRegression"):
        forecast = clfreg.predict(X_lately)
        dfreg['LinearRegression'] = np.nan
    if (model == "Polynomial2"):
        forecast = clfpoly2.predict(X_lately)
        dfreg['Polynomial2'] = np.nan
    if (model == "Polynomial3"):
        forecast = clfpoly3.predict(X_lately)
        dfreg['Polynomial3'] = np.nan
    if (model == "LassoCV"):
        forecast = clfLasso.predict(X_lately)
        dfreg['LassoCV'] = np.nan
    
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]


#вивід на графік
plt.plot(dfreg['Adj Close'].tail(forecast_out*6), label="Adj Close")
plt.plot(dfreg['LinearRegression'].shift(-2*forecast_out), label="LinearRegression")
plt.plot(dfreg['Polynomial2'].shift(-3*forecast_out), label="Polynomial2")
plt.plot(dfreg['Polynomial3'].shift(-4*forecast_out), label="Polynomial3")
plt.plot(dfreg['LassoCV'].shift(-5*forecast_out), label="LassoCV")

plt.legend(loc=4)
plt.title("Model comparison")
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
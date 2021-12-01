# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:59:41 2021

@author: Asus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

datas = pd.read_csv('archive/apartment_prices.csv').sort_values('Squaremeter')

print("Shape :",datas.shape)
squaremeter = datas.iloc[:,:1].values
price = datas.iloc[:,-1].values
#%%--------------------------------- feature Engineering-----------------------
print(datas.sample(5))

# plot the distribution plots for the fetures
plt.figure(figsize = (16,5))

plt.subplot(1,2,1)
sns.distplot(datas['Squaremeter'])
plt.subplot(1,2,2)
sns.distplot(datas['Price'])
plt.show()

#%%Boundary Values
"""
print("Higest allowed", datas['Squaremeter'].mean()+3*datas['Squaremeter'].std())
print("Lowest allowed", datas['Squaremeter'].mean()-3*datas['Squaremeter'].std())
"""
print("Higest allowed", datas['Price'].mean()+2*datas['Price'].std())
print("Lowest allowed", datas['Price'].mean()-2*datas['Price'].std())

print(datas[(datas['Price']> 537.28) | (datas['Price']< 70.91)])
#print(datas[(datas['Squaremeter']> 119.56) | (datas['Squaremeter']< 5.11)])

#%% clean datas
new_datas = datas[(datas['Price']< 537.28) & (datas['Price']> 70.91)]

print("Shape :",new_datas.shape)
squaremeter = new_datas.iloc[:,:1].values
price = new_datas.iloc[:,-1].values

#%% Data Statistic
print(datas['Price'].describe())
#%% Box plot the distribution plot for the features
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(datas['Price'])
plt.subplot(1,2,2)
sns.distplot(datas['Squaremeter'])
plt.show()

plt.figure(figsize=(16,5))
sns.boxplot(datas['Squaremeter'])
plt.show()
#%% scale number

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
datas_scale = sc.fit_transform(new_datas)

squaremeter_sc = datas_scale[:,0:1]
price_sc = datas_scale[:,-1]

#%% plot 
print(str(squaremeter.min()))
print(str(squaremeter.max()))
print(str(price.max()))
plt.plot(price,squaremeter,color='blue')
plt.xlabel('squaremeter')
plt.ylabel('price')
plt.show()

#%% corallation

plt.figure(figsize=(10,5))
corr_matrix = datas.corr()
sns.heatmap(corr_matrix,annot=True)
plt.show()

#%%----------------------------------train test split--------------------------

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(squaremeter,price, test_size=0.33,random_state = 0)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
#%% scale train test split

X_train_sc,X_test_sc,y_train_sc,y_test_sc = train_test_split(squaremeter_sc,price_sc, test_size=0.33,random_state = 0)

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#%% --------------------------------prediction linear algo---------------------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
price_reg = lr.fit(X_train, y_train)

predict_price = price_reg.predict(X_test)
#%% scale  data Linear algo
price_reg_sc = lr.fit(X_train_sc, y_train_sc)

predict_price_sc = price_reg.predict(X_test_sc)



#%% -------------------------------logistic regression-------------------------
from sklearn.linear_model import LogisticRegression

# scale edilmemiş
lr = LogisticRegression()
lr_price_reg = lr.fit(X_train,y_train)
lr_predict_reg = lr_price_reg.predict(X_test)



#%%----------------------------------Decision Tree-----------------------------
from sklearn.tree import DecisionTreeRegressor

dtr_clf = DecisionTreeRegressor()
dtr_clf = dtr_clf.fit(X_train,y_train)
dtr_clf_predict = dtr_clf.predict(X_test)

#%%-------------------------------Support Vector machine-----------------------
from sklearn.svm import SVR

svr_r = SVR(kernel='linear',epsilon=0.23)
svr_price = svr_r.fit(X_train, y_train)
svr_price_predict = svr_price.predict(X_test)

#%%------------------------------Polynomial Regression-------------------------
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
poly_squaremeter = poly_reg.fit_transform(X_train)
lr_2 = LinearRegression()
lr_2.fit(poly_squaremeter,y_train)
poly_predict = lr_2.predict(poly_reg.fit_transform(X_test))
plt.scatter(X_test,y_test)
plt.plot(X_test,lr_2.predict(poly_reg.fit_transform(X_test)))
plt.show()

#%%-----------------------------Random Forest Regression-----------------------

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state = 0)
rfr.fit(X_train,y_train)
rfr_predict = rfr.predict(X_test)



#%%  R^2  Regression Result
from sklearn.metrics import r2_score
print('Linear Regression R2 degeri')
print(r2_score(y_test, predict_price))
print('Linear Regression R2 degeri scale data')
print(r2_score(y_test_sc, predict_price_sc))
print('Logistic Regression R2 degeri')
print(r2_score(y_test,lr_predict_reg))
print('Decision Tree Regression R2 degeri')
print(r2_score(y_test,dtr_clf_predict))
print('Support Vector Regression R2 degeri')
print(r2_score(y_test,svr_price_predict))
print('Polynomial Regression R2 degeri')
print(r2_score(y_test,poly_predict))
print('Random Forest Regression R2 degeri')
print(r2_score(y_test,rfr_predict))

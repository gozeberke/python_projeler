# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:53:11 2021

@author: acer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



veri=pd.read_csv("tenis.csv")
#kategorik veriyi dönüştürme

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

hava_durumu=veri.iloc[:,0:1].values
hava_durumu[:,0]=le.fit_transform(veri.iloc[:,0])
ohe=preprocessing.OneHotEncoder()
hava_durumu=ohe.fit_transform(hava_durumu).toarray()
print(hava_durumu)

play=veri.iloc[:,-1:].values
play[:,0]=le.fit_transform(veri.iloc[:,-1:])
play=ohe.fit_transform(play).toarray()
print(play)

windy=veri.iloc[:,3:4].values
windy[:,0]=le.fit_transform(veri.iloc[:,3:4])
windy=ohe.fit_transform(windy).toarray()
print(windy)

#---------kukla değişkenleri atma

ywindy=windy[:,1:]
yplay=play[:,1:]
sonuc4=pd.DataFrame(data=yplay,index=range(14),columns=["play"])
sonuc3=pd.DataFrame(data=ywindy,index=range(14),columns=["windy"])
sıcaklık=veri.iloc[:,1:2].values
sonuc2=pd.DataFrame(data=sıcaklık,index=range(14),columns=["temperature"])
sonuc=pd.DataFrame(data=hava_durumu,index=range(14),columns=["overcast","rainy","sunny"])
s1=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([sonuc3,sonuc4],axis=1)
s3=pd.concat([s1,s2],axis=1)

humidity=veri.iloc[:,2:3].values
humidity=pd.DataFrame(data=humidity,index=range(14),columns=['humidity'])
#-----------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s3,humidity,test_size=0.33,random_state=0 )
#_____ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
print(y_pred)

# Geri eleme 
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=s3,axis=1)
X_l=s3.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(humidity,X_l).fit()
print(model.summary())

X=np.append(arr=np.ones((14,1)).astype(int),values=s3,axis=1)
X_l=s3.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(humidity,X_l).fit()
print(model.summary())


X=np.append(arr=np.ones((14,1)).astype(int),values=s3,axis=1)
X_l=s3.iloc[:,[1,3]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(humidity,X_l).fit()
print(model.summary())


x_train,x_test,y_train,y_test=train_test_split(X_l,humidity,test_size=0.33,random_state=0 )
regressor2=LinearRegression()
regressor2.fit(x_train,y_train)

y_pred=regressor2.predict(x_test)
print(y_pred)
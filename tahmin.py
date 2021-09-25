# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:45:22 2021

@author: acer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Aylara göre satışların tahmin edilmesi

veriler=pd.read_csv("aylaragoresatis.csv")
aylar=veriler[['Aylar']]
satislar=veriler[['Satislar']]
#verilerin test ve train olarak ayrılması
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#verilerin kendi arasında ölçeklenmesi
'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
'''

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin =lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara göre satis")
plt.xlabel("Aylar")
plt.ylabel("Satislar")
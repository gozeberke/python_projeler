# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:02:44 2021

@author: acer
"""
import numpy as np 
import pandas as pd

veri=pd.read_csv("veriler.csv")

boykiloyas=veri.iloc[:,1:4].values
cinsiyet=veri.iloc[:,-1:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(boykiloyas,cinsiyet,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski')
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)

from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('SVC --linear')
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


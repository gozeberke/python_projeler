# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 16:19:11 2021

@author: acer
"""
#boy , yas ,kilo ve ülke bilgilerinden cinsiyet çıkarımı
import pandas as pd 
import numpy as np
veri=pd.read_csv("veriler.csv")


from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veri.iloc[:,1:4].values
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke=veri.iloc[:,0:1].values
ulke[:,0]=le.fit_transform(veri.iloc[:,0])
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

cinsiyet=veri.iloc[:,-1:].values
cinsiyet[:,0]=le.fit_transform(veri.iloc[:,-1:])
cinsiyet=ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
#cinsiyet=veri.iloc[:,-1].values
sonuc3=pd.DataFrame(data=cinsiyet[:,1:],index=range(22),columns=["cinsiyet"])
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0 )

#verilerin kendi içerisinde ölçeklenmesi 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

boy=s2.iloc[:,3:4].values
print(boy)

sol=s2.iloc[:,:3].values
sol=pd.DataFrame(data=sol,index=range(22),columns=['fr','tr','us'])
sag=s2.iloc[:,4:].values
sag=pd.DataFrame(data=sag,index=range(22),columns=['kilo','yas','cinsiyet'])
yeniveri=pd.concat([sol,sag],axis=1)
x_train,x_test,y_train,y_test=train_test_split(yeniveri,boy,test_size=0.33,random_state=0 )



regressor2=LinearRegression()
regressor2.fit(x_train,y_train)

y_pred=regressor2.predict(x_test)
# Geri eleme 
import statsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=yeniveri,axis=1)
X_l=yeniveri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=yeniveri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=yeniveri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())
#verinin düzenlenmiş hali ile tahmin prediction
x_train,x_test,y_train,y_test=train_test_split(X_l,boy,test_size=0.33,random_state=0 )
regressor2=LinearRegression()
regressor2.fit(x_train,y_train)

y_pred=regressor2.predict(x_test)
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:17:27 2021

@author: acer
"""

import pandas as pd 
import numpy as np
veri=pd.read_csv("eksikveri.csv")
# eksik verileri düzenleme



from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veri.iloc[:,1:4].values
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)
#kategorik verileri nümerik veriye dönüştürme



from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke=veri.iloc[:,0:1].values
ulke[:,0]=le.fit_transform(veri.iloc[:,0])
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
#eksik verilerin ve kategorik verilerin düzenlenmiş halinin birleştirilmesi




sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
cinsiyet=veri.iloc[:,-1].values
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)
#verinin test ve train olarak bölünmesi


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0 )

#verilerin kendi içerisinde ölçeklenmesi 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)




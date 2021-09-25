

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score


veri=pd.read_csv("maaslar_yeni.csv")
maas=veri.iloc[:,5:]
x=veri.iloc[:,2:3]
Maas=maas.values
X=x.values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Maas) 
# Geri eleme 

model=sm.OLS(lin_reg.predict(X),x)
print(model.fit().summary())
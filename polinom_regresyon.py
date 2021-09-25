import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("maaslar.csv")

egitim=veriler.iloc[:,1:2].values
maas=veriler.iloc[:,2:].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(egitim,maas) #xten y öğren

plt.scatter(egitim,maas,color='red')
plt.plot(egitim,lin_reg.predict(egitim),color='blue')
plt.show()

#polinom regresyon
#2.dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
egitim_poly=poly_reg.fit_transform(egitim)
print(egitim_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(egitim_poly,maas)
plt.scatter(egitim,maas,color='red')
plt.plot(egitim,lin_reg2.predict(poly_reg.fit_transform(egitim)),color='blue')
plt.show()
#4.dereceden
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4 )
egitim_poly=poly_reg.fit_transform(egitim)
print(egitim_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(egitim_poly,maas)
plt.scatter(egitim,maas,color='red')
plt.plot(egitim,lin_reg2.predict(poly_reg.fit_transform(egitim)),color='blue')
plt.show()

#tahmin
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
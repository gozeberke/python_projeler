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

#SVR kullanmak için scale lazım
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()

egitim_olcekli=sc1.fit_transform(egitim)
sc2=StandardScaler()
maas_olcekli=sc2.fit_transform(maas)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(egitim_olcekli,maas_olcekli)
plt.scatter(egitim_olcekli,maas_olcekli,color='red')
plt.plot(egitim_olcekli,svr_reg.predict(egitim_olcekli),color='blue')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

veriler=pd.read_csv("maaslar.csv")

egitim=veriler.iloc[:,1:2].values
maas=veriler.iloc[:,2:].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(egitim,maas) #xten y öğren
print("Liner Regresyon Grafik")
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
print("2.Dereceden Polinom Regresyon Grafik")

plt.scatter(egitim,maas,color='red')
plt.plot(egitim,lin_reg2.predict(poly_reg.fit_transform(egitim)),color='blue')
plt.show()
#4.dereceden polinom regresyon
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4 )
egitim_poly=poly_reg.fit_transform(egitim)
print(egitim_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(egitim_poly,maas)
print("4. Dereceden Polinom Regresyon Grafik")
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
#Destek Vektör Regresyon
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(egitim_olcekli,maas_olcekli)
print("Destek Vektör Regresyon Grafik")
plt.scatter(egitim_olcekli,maas_olcekli,color='red')
plt.plot(egitim_olcekli,svr_reg.predict(egitim_olcekli),color='blue')
plt.show()

#Karar Agacı Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(egitim,maas)
print("Karar Agacı Regresyon Grafik")
plt.scatter(egitim,maas,color='red')
plt.plot(egitim,r_dt.predict(egitim),color='blue')
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

#Rassal Ağaç Regresyon

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(egitim,maas.ravel())
print(rf_reg.predict([[6.6]]))
print("Rassal Agac Regresyon Grafik")

plt.scatter(egitim,maas,color='red')
plt.plot(egitim,rf_reg.predict(egitim),color='blue')
plt.show()
# R2 sonuçları
print('Random forest r2 sonucu:')
print(r2_score(maas,rf_reg.predict(egitim)))
print('------------------------------')
print('Karar Agacı r2 sonucu:')
print(r2_score(maas,r_dt.predict(egitim)))
print('------------------------------')
print('Destek Vektör  r2 sonucu:')
print(r2_score(maas_olcekli,svr_reg.predict(egitim_olcekli)))
print('------------------------------')
print('Polinom r2 sonucu:')
print(r2_score(maas,lin_reg2.predict(poly_reg.fit_transform(egitim))))
print('------------------------------')
print('Linear r2 sonucu:')
print(r2_score(maas,lin_reg.predict(egitim)))



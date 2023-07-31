import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn import linear_model

#downloading dataset 
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
         with open(filename, "wb") as f:
             f.write(response.content)
    
download(path, "FuelConsumption.csv")
path="FuelConsumption.csv"


df = pd.read_csv("FuelConsumption.csv")


cdf = df[['FUELCONSUMPTION_CITY','ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','FUELCONSUMPTION_HWY','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


#modelling
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','ENGINESIZE', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print ('Coefficients: ', regr.coef_)
# print ('Intercept: ',regr.intercept_)


plt.scatter(train.CYLINDERS, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

test_x = np.asanyarray(test[['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','ENGINESIZE', 'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Residual sum of squares: %.2f"
      % np.mean((test_y_ - test_y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_x, test_y))
 







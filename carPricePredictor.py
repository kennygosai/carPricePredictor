import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import sys

if len(sys.argv) != 4:
    print("Need 4 arguments: python carPricePredictor.py <Mileage> <Cylinder> <Doors>")
    

print ("Number of arguments: ", len(sys.argv))
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']] #feature variables
y = df['Price'] #trying to predict

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']]) #scales values into uniform ranges between negative 1 and positive 1

# print (X) Print data 

est = sm.OLS(y, X).fit() #creates function which is going to be used for predicting the car values.

# print(est.summary()) print summary after. Shows coefficients of the regression are :
# Mileage    -1272.3412    
# Cylinder    5587.4472   
# Doors      -1404.5513

# print(y.groupby(df.Doors).mean()) Shows that doors does not really affect the cost of the car with 2 doors and 4 doors being relatively the same price
# Doors
# 2    23807.135520
# 4    20580.670749

# scaled = scale.transform([[45000, 8, 4]])
scaled = scale.transform([[sys.argv[1], sys.argv[2], sys.argv[3]]])
# print(scaled) prints the scaled version
predicted = est.predict(scaled[0])
print(predicted)
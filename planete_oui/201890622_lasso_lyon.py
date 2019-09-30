# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')

import datetime as dt
from datetime import datetime
import calendar

dataInt['consumption_1'] = dataOut['consumption_1']
dataInt['consumption_2'] = dataOut['consumption_2']
print(dataInt.dtypes)
dataInt["date"] = dataInt.timestamp.apply(lambda x : x.split('T')[0])
dataInt["hour"] = dataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
dataInt['date'] = pd.to_datetime(dataInt['date']).dt.strftime("%Y%m%d").astype(int)


print(dataInt.dtypes)
#categoryVariableList = ["loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"]
#for var in categoryVariableList:
#   dataInt[var] = dataInt[var].astype("category")
dataInt = dataInt.drop(["timestamp", "ID", "loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"],axis=1)


cols = dataInt.columns.tolist()
dataInt = dataInt[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2','consumption_secondary_1', 
                   'consumption_secondary_2','consumption_secondary_3', 'date', 'hour', 
                   'consumption_1', 'consumption_2']]

print("dataInt:\n", dataInt.isnull().sum())
print(dataInt.dtypes)
dataInt = dataInt.astype(float)

X = dataInt.iloc[:, :-2].values.astype(float)
y = dataInt.iloc[:, -2:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])

z = np.append(X, y, axis=1)
from sklearn.model_selection import train_test_split
z_train,z_test = train_test_split(z, test_size = 0.2)


# REGRESSION LINEAIRE MULTIPLE pour la consomation du point 1


#XTrain = z_train.iloc[:,:-]
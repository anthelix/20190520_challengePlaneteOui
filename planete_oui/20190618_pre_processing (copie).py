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

dataInt["date"] = dataInt.timestamp.apply(lambda x : x.split('T')[0])
dataInt["hour"] = dataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
#dataInt["weekday"] = dataInt.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
#dataInt["month"] = dataInt.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

dataInt['date'] = pd.to_datetime(dataInt['date']).dt.strftime("%Y%m%d").astype(int)
categoryVariableList = ["loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"]
for var in categoryVariableList:
    dataInt[var] = dataInt[var].astype("category")

    
dataInt = dataInt.drop(["timestamp", "ID", "loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"],axis=1)


X = dataInt.iloc[:, 1:].values
y = dataOut.iloc[:, 1:].values

print("dataInt:\n", dataInt.isnull().sum())
print(dataInt.dtypes)


# Taking care of missing data
# voir pour prendre les donnees sur une periode et non sur tout le dataset 
# ou voir pour 'most frequent' sur une periode 
# ou regression
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, :5])
X[:, :5] = imputer.transform(X[:, :5])





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# construction du modele multiple
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#  Faire de nouvelles predictions
y_pred  = regressor.predict(X_test) # valeur des VI dont on peut predire la valeur des VD
y_pred # predictions a partir de valeurs dans le test set que nous avonc construit
regressor.predict(np.array([[1, 0, 130000, 140000, 300000]]))# predictions vous VIqui ne sont pas dans le X_dataset

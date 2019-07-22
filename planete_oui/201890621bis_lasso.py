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

dataInt["date"] = dataInt.timestamp.apply(lambda x : x.split('T')[0])
dataInt["hour"] = dataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
#dataInt["weekday"] = dataInt.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
#dataInt["month"] = dataInt.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

dataInt['date'] = pd.to_datetime(dataInt['date']).dt.strftime("%Y%m%d").astype(int)
categoryVariableList = ["loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"]
for var in categoryVariableList:
    dataInt[var] = dataInt[var].astype("category")

    
dataInt = dataInt.drop(["timestamp", "ID", "loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"],axis=1)


cols = dataInt.columns.tolist()
dataInt = dataInt[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2','consumption_secondary_1', 
                   'consumption_secondary_2','consumption_secondary_3', 'date', 'hour', 
                   'consumption_1', 'consumption_2']]

print("dataInt:\n", dataInt.isnull().sum())
print(dataInt.dtypes)


X = dataInt.iloc[:, :-2].values
y = dataInt.iloc[:, -2:].values




# Taking care of missing data
# voir pour prendre les donnees sur une periode et non sur tout le dataset 
# ou voir pour 'most frequent' sur une periode 
# ou regression
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# construction du modele multiple
from sklearn.metrics import r2_score


# #############################################################################
# Lasso
from sklearn.linear_model import Lasso
alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

print(lasso.coef_)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

#centrer et réduire les données d'apprentissage


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(dataInt, test_size = 0.2)
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(data_train[:, 0:6])
data_train[:, 0:6] = imputer.transform(data_train[:, 0:6])



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Ztrain =sc.fit_transform(X_train)
print(np.mean(Ztrain, axis=0))
print(np.var(Ztrain, axis=0))

from sklearn.linear_model import Lasso
regLasso1 = Lasso(fit_intercept=False,normalize=False)
print(regLasso1)
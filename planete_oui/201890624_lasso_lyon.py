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

dataInt.head()
dataInt.dtypes
dataInt.info()
dataInt.describe()
dataInt_Xy = dataInt.copy()

dataInt_Xy['timestamp']=pd.to_datetime(dataInt_Xy['timestamp']) 

def conv(data):
    data['year']=data['timestamp'].dt.year
    data['month']=data['timestamp'].dt.month
    data['day']=data['timestamp'].dt.day
    data['hour']=data['timestamp'].dt.hour
    #data['sec']=data['timestamp'].dt.second
    #data['min']=data['timestamp'].dt.minute
    return data

dataInt_Xy = conv(dataInt_Xy)


dataInt_Xy['consumption_1'] = dataOut['consumption_1']
dataInt_Xy['consumption_2'] = dataOut['consumption_2']
#cols = dataInt.columns.tolist()
dataInt_Xy = dataInt_Xy[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2',
                   'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3',
                   "loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3",
                   'year', 'month', 'day','hour', 
                   'consumption_1', 'consumption_2']]
print("dataInt:\n", dataInt.isnull().sum())
dataInt_Xy_consum=dataInt_Xy.groupby(['year','month'])['consumption_1', 'consumption_2' ].sum().reset_index()

# split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataInt_Xy, test_size=0.2)
test_set.head()


#dataInt["date"] = dataInt.timestamp.apply(lambda x : x.split('T')[0])
#dataInt["hour"] = dataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
#dataInt['date'] = pd.to_datetime(dataInt['date']).dt.strftime("%Y%m%d").astype(int)
#dataInt["weekday"] = dataInt.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
#dataInt["month"] = dataInt.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
consum = train_set.drop(['consumption_1','consumption_2'], axis=1)
consum_labels = train_set.iloc[:, -2:]

sample_incomplete_rows = consum[consum.isnull().any(axis=1)].head()

try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
imputer = SimpleImputer(strategy="median")
# remove none numbers
consum_num = consum.select_dtypes(include=[np.number])
imputer.fit(consum_num)
imputer.statistics_
consum_num.median().values

X = imputer.transform(consum_num)
consum_tr = pd.DataFrame(X, columns=consum_num.columns,
                          index = list(consum.index.values))

consum_tr.loc[sample_incomplete_rows.index.values]
consum_tr = pd.DataFrame(X, columns=consum_num.columns)



consum_cat = housing[['ocean_proximity']]


print(dataInt.dtypes)
#categoryVariableList = ["loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3"]
#for var in categoryVariableList:
#    dataInt[var] = dataInt[var].astype("category")
#dataInt = dataInt.drop(["timestamp"],axis=1)





print(dataInt.dtypes)


print("dataInt:\n", consum.isnull().sum())

#dataInt = dataInt.astype(float)




# PARTITION APPRENTISSAGE TEST ET GESTION DES MISSING VALUES
X = dataInt.iloc[:, :-2].values
#.astype(float)
y = dataInt.iloc[:, -2:].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])
print(X)

z = np.append(X, y, axis=1)


from sklearn.model_selection import train_test_split
z_train,z_test = train_test_split(z, test_size = 0.2)
print(z_train.shape)
print(z_test.shape)


# REGRESSION LINEAIRE MULTIPLE pour la consomation du point 1
XTrain = z_train.iloc[:,:-2]










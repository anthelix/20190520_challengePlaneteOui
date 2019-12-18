### changement par rapport a pdbdm
#mise sous une fonction les modofications et ajout de colonne
#agregation  dans pdbdm de dataInt et DataOUt pour avoir une meilleur moyenne sur les missing values

#importing the libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from vacances_scolaires_france import SchoolHolidayDates
import sys
from impyute.imputation.cs import mice
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#pip install vacances-scolaires-france
#pip3 install impyute
# my fonctions
# my fonctions

def conv(data):
    data["date"] = data.timestamp.apply(lambda x : x.split('T')[0])
    data["datetime_perso"] = data.timestamp.apply(lambda x : get_format_the_date(x))
    data['year']=data['datetime_perso'].dt.year
    data['month']=data['datetime_perso'].dt.month
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hours'] = data['timestamp'].dt.hour
    return data

## get season
def get_season(doy):
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    if doy in spring:
        season = 'spring'
    elif doy in summer:
      season = 'summer'
    elif doy in fall:
        season = 'fall'
    else:
        season = 'winter'
    return season

## verifie si jour ferie
def easter_date(year):
    a = year // 100
    b = year % 100
    c = (3 * (a + 25)) // 4
    d = (3 * (a + 25)) % 4
    e = (8 * (a + 11)) // 25
    f = (5 * a + b) % 19
    g = (19 * f + c - e) % 30
    h = (f + 11 * g) // 319
    j = (60 * (5 - d) + b) // 4
    k = (60 * (5 - d) + b) % 4
    m = (2 * j - k - g + h) % 7
    n = (g - h + m + 114) // 31
    p = (g - h + m + 114) % 31
    day = p + 1
    month = n
    return datetime(year, month, day)

def is_ferie(the_date):
    year = the_date.year
    easter = easter_date(year)
    days = [
        datetime(year, 1, 1),  # Premier de l'an
        easter + timedelta(days=1),  # Lundi de Pâques
        datetime(year, 5, 1),  # Fête du Travail
        datetime(year, 5, 8),  # Victoire de 1945
        easter + timedelta(days=39),  # Ascension
        easter + timedelta(days=49),  # Pentecôte
        datetime(year, 7, 14),  # Fête Nationale
        datetime(year, 8, 15),  # Assomption
        datetime(year, 11, 1),  # Toussaint
        datetime(year, 11, 11),  # Armistice 1918
        datetime(year, 12, 25),  # Noël
    ]
    return the_date in days

def get_format_the_date(timestamp):
    do = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    d5 = do.replace(minute=0, hour=0, second=0, microsecond=0).isoformat(' ')
    the_date = datetime.strptime(d5, "%Y-%m-%d %H:%M:%S")
    return the_date

def business_day(timestamp):
    if not is_ferie(timestamp) and timestamp.isoweekday() not in [6, 7]:
        return True
    else:
        return False
    
    
#creating a function to encapsulate preprocessing, to mkae it easy to replicate on  submission data
# tester aussi avec dataInt et DataTeat separer dans une future version

def processing(dataInt):
    ## missing value
    df = dataInt.copy()
    df_num = df.drop(['timestamp','loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
    df_NonNum = df.select_dtypes(include=[np.object])
    imputed_training_mice = mice(df_num.values)
    data_mice = pd.DataFrame(imputed_training_mice, columns = df_num.columns, index = list(df.index.values))
    dClean = data_mice.join(df_NonNum)
    ## drop variable inutile
    d_tr = dClean.drop(['loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
    ## create extra attribute
    conv(d_tr)
    d_tr['timestamp'] = pd.to_datetime(d_tr.timestamp, format = '%Y-%m-%dT%H:%M:%S.%f')
    ## create season and rangeInYear
    s = pd.to_datetime(pd.Series(d_tr['timestamp']))
    d_tr['rangeInYear'] = s.dt.strftime('%j').astype(int)
    d_tr['season'] = d_tr['rangeInYear'].apply(lambda d : get_season(d))
    #create jours working days
    d_tr['is_business_day'] = d_tr['datetime_perso'].apply(lambda e : int(business_day(e)))
    # Is it an holiday for zone A, B or C?
    d = SchoolHolidayDates()
    d_tr['is_holiday'] = d_tr['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))

    dataInt1 = d_tr.drop(['rangeInYear', 'datetime_perso', 'date', 'timestamp'], axis=1)
    return (dataInt1)    

#------------------------------------------------------------------------------    

# creere un je de test
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')
data_blink = pd.concat([dataInt, dataOut[['consumption_1', 'consumption_2']]], axis=1)

#----------------------
dataInt = dataInt.drop(['ID'], axis=1)
dataOut = dataOut.drop(['ID'], axis=1)


#------------------------------------------------------------------------------
#SPLIT
tscv = TimeSeriesSplit(n_splits=10)
print(tscv)
for train_index, test_index in tscv.split(dataInt):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataInt.iloc[train_index, :], dataInt.iloc[test_index, :]
    y_train, y_test = dataOut.iloc[train_index, :], dataOut.iloc[test_index, :]

y_train = y_train.values.reshape(-1, 2)
y_test = y_test.values.reshape(-1,2)
y_train.shape

#------------------------------------------------------------------------------


#--------------------------TRAIN-----------------------------------------------



#----------------------
Xtrain = processing(X_train)
Xtrain.head()
Xtrain.columns



numerical_features = [f for f in Xtrain.columns if Xtrain[f].dtype == float]
Xtrain_num = Xtrain[numerical_features].values
scaler =  StandardScaler()
Xtrain_num = scaler.fit_transform(Xtrain_num)
#^^^^^^^^^
binary_features = ['is_holiday','is_business_day']


categorical_features = ['year', 'month', 'hours', 'season' ]
for var in categorical_features:
    Xtrain[var] = Xtrain[var].astype('category')
Xtrain_cat = Xtrain[categorical_features].values

encoder = OneHotEncoder()
encoder.categories_
encoder.get_feature_names()



Xtrain_cat = pd.DataFrame(Xtrain_cat)
Xtrain_cat.head(5)
Xtrain_cat.info()
Xtrain_cat.drop(['x0_2016', 'x1_1','x2_0', 'x3_fall'], inplace=True, axis=1)
#^^^^^^^^^

#Get Feature Names of Encoded columns
encoder.get_feature_names()
Xtrain_cat
Xtrain_cat.shape
type(Xtrain_cat)



# Converting the numpy array into a pandas dataframe
d_encoded_data2 = pd.DataFrame(Xtrain_cat, columns=encoder.get_feature_names())
d_encoded_data.drop(['oh_enc__x0_2016', 'oh_enc__x1_1','oh_enc__x2_0', 'oh_enc__x3_0','oh_enc__x4_0', 'oh_enc__x5_fall'], inplace=True, axis=1)
#Concatenating the encoded dataframe with the original dataframe
df_concat = pd.concat([d_ready.reset_index(drop=True), d_encoded_data.reset_index(drop=True)], axis=1)
# Dropping drive-wheels, make and engine-location columns as they are encoded
df_concat.drop(['season', 'year', 'month', 'hours', 'is_business_day', 'is_holiday'], inplace=True, axis=1)
# Viewing few rows of data
df_concat.info()



#??????????????????????????????????????????????????????????????????????????????
Xtrain_new.shape
Xtrain_new.dtypes
type(col_name_train)
Xtrain_new[col_name_train].dtypes
Xtrain_new[col_name_train].dtypes
type(numeric_features_train)
Xtrain_new.info()
dataInt.dtypes
# jeu de donnees propres:
    # X-train et y_train pour faire le modele
    # X_test et y_test pour tester mon modele
    # dataTest pour la soumission


#----------------------------TEST----------------------------------------------
Xtest = processing(X_test)
for var in categorical_features:
    Xtest[var] = Xtest[var].astype('category')
numerical_features = [f for f in Xtest.columns if Xtest[f].dtype == float]
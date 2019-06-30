#importing the libraries
import numpy as np
import missingno as msno  # missing value
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta, time
import calendar
from vacances_scolaires_france import SchoolHolidayDates
import sys
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice
#from fancyimpute import MICE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# my fonctions
def conv(data):
    data["date"] = data.timestamp.apply(lambda x : x.split('T')[0])

    # the hours and if it's night or day (7:00-22:00)
    
    

    #data["hour"] = data.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).hour
    #data['hour'] = data['hour'].dt.hour
    #data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
    #data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
    data["datetime_perso"] = data.timestamp.apply(lambda x : get_format_the_date(x))
    data['year']=data['datetime_perso'].dt.year
    data['month']=data['datetime_perso'].dt.month
    data['weekday']=data['datetime_perso'].dt.day
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hours'] = data['timestamp'].dt.hour
    #data['hour']=data['datetime_perso'].dt.hour
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
    """
    Calcule la date du jour de Pâques d'une année donnée
    Voir https://github.com/dateutil/dateutil/blob/master/dateutil/easter.py
    
    :return: datetime
    """
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
    """
    Vérifie si la date donnée est un jour férié
    :param the_date: datetime
    :return: bool
    """
    #the_date = get_format_the_date(the_date)
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
    
#------------------------------------------------------------------------------    
# creere un je de test
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')
dataInt.info()

#combine
data_raw = dataInt.append(dataTest)
data = pd.merge(data_raw, dataOut, on='ID', how='left')
#data engeebering

dI = data.copy()
dI_labels = dI.drop(['ID', 'timestamp', 'temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3', 'loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
dI_num = dI.drop(['timestamp','loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
imputed_training_mice=mice(dI_num.values)
data_mice = pd.DataFrame(imputed_training_mice, columns=dI_num.columns, index = list(dI.index.values))
dI_NonNum = dI.select_dtypes(include=[np.object])
dClean = data_mice.join(dI_NonNum)
d_tr = dClean.drop(['loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)

#create extra attribute

d_tr.info()
conv(d_tr)
#d_tr['day of week']=d_tr['datetime_perso'].dt.dayofweek 


d_tr['timestamp'] = pd.to_datetime(d_tr.timestamp, format = '%Y-%m-%dT%H:%M:%S.%f')
## create season and rangeInYear
s = pd.to_datetime(pd.Series(d_tr['timestamp']))
d_tr['rangeInYear'] = s.dt.strftime('%j').astype(int)
d_tr['season'] = d_tr['rangeInYear'].apply(lambda d : get_season(d))
## create jours working days

d_tr['is_business_day'] = d_tr['datetime_perso'].apply(lambda e : int(business_day(e)))

# Is it an holiday for zone A, B or C?
d = SchoolHolidayDates()
d_tr['is_holiday'] = d_tr['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))
d_tr = d_tr.drop(['rangeInYear', 'datetime_perso', 'date'], axis=1)


d_tr.info()


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# Gerer les variables categoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
features = ['season', 'is_business_day', 'is_holiday']

ct = ColumnTransformer([('oh_enc', OneHotEncoder(sparse=False), [16, 17, 18])], remainder='passthrough')
d_tr = df.DataFrame(ct.fit_transform(d_tr))

d_tr['season'].values

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X

# enlever relation relation d'ordre en creant colomnnes
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

   





categorical_attributes = list(d_tr.select_dtypes(include=['object', 'bool']).columns)
numerical_attributes = list(d_tr.select_dtypes(include=['float64', 'int64']).columns)


categorical_features = features.dtypes == 'float', 'int'
numeriacal_features = ~categorical_features



dClean.isnull().sum() # A ENLEVER
#-------------------------------------------------------------------------

# IMPORTING THE DATA SET AND COMBINE
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')
dataInt.info()
#combine Input train, Input Test and  output Train
data_raw = dataInt.append(dataTest)
data = pd.merge(data_raw, dataOut, on='ID', how='left')


data = data.drop('ID', axis=1)
## CONSTRUCTION DU MODELE


# Feature Enginering

# creating New columns from 'timestamp'
conv(data)




data['day of week']=data['datetime_perso'].dt.dayofweek 
temp = data['datetime_perso']
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
temp2 = data['datetime_perso'].apply(applyer) 
data['weekend']=temp2

## create season and rangeInYear
s = pd.to_datetime(pd.Series(data['timestamp']))
data['rangeInYear'] = s.dt.strftime('%j').astype(int)
data['season'] = data['rangeInYear'].apply(lambda d : get_season(d))



## create jours working days
data['is_business_day'] = data['datetime_perso'].apply(lambda e : int(business_day(e)))

# Is it an holiday for zone A, B or C?
d = SchoolHolidayDates()
data['is_holiday'] = data['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))

data.info()
## missing values
# matrice des donnees manquantes
data.isnull().sum() # A ENLEVER

#data_num = [['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2', 
    #'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3', 
    #'day of week', 'weekend', 'rangeInYear', 'is_business_day', 'is_holiday']]
#data_numNoNeedPredic = [[ 'day of week', 'weekend', 'rangeInYear', 'is_business_day', 'is_holiday']]
data_raw_num = data_raw[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']]

#data_obj = data[['timestamp', 'loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 
    #'loc_secondary_3', 'date', 'hour', 'weekday', 'month', 'season']]
data_labels = data[['consumption_1', 'consumption_2']]
# join data_mice avec data_obj+data_numNoNeedPredic+data_labels

data_nonePredict = data[['timestamp','loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3','date', 'hour', 'weekday', 'month', 'season','day of week', 'weekend', 'rangeInYear', 'is_business_day', 'is_holiday','consumption_1', 'consumption_2']]
#sample_incomplete_rows = data_num[data_num.isnull().any(axis=1)]

#categorical_attributes = list(data.select_dtypes(include=['object']).columns)
#numerical_attributes = list(dataInt.select_dtypes(include=['float64', 'int64']).columns)
#numerical_attributes = list(data.select_dtypes(include=['float64', 'int64']).columns)
#print('categorical_attributes:', categorical_attributes)
#print('numerical_attributes:', numerical_attributes)

# imputation par MICE
imputed_training_mice=mice(data_raw_num.values)
data_mice = pd.DataFrame(imputed_training_mice, columns=data_raw_num.columns, index = list(data.index.values))
data_clean = data_mice.join(data_nonePredict)

data_clean.isnull().sum() # A ENLEVER


# Imputation par KNN 
#sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
# start the KNN training
#imputed_training_KNN=fast_knn(data_missing.values, k=30)



# Coercing To Category Type
categoricalFeatureNames = ['date', 'hour', 'weekday', 'month', 'season', 'is_business_day', 'is_holiday']
numericalFeatureNames = ['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2', 'consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3']
dropFeatures = ['timestamp','rangeInYear', 'day of week', 'weekend', 'loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3']
label = ['consumption_1', 'consumption_2']
for var in categoricalFeatureNames:
    data_clean[var] = data_clean[var].astype("category")

# Splitting Train And Test Data
#dataTrain = data_clean[pd.notnull(data_clean['consumption_1'])].sort_values(by=['timestamp'])
#dataTest = data_clean[~pd.notnull(data_clean['consumption_1'])].sort_values(by=['timestamp'])
#datatimecol = dataTest['timestamp']
yLabels = dataTrain[['consumption_1', 'consumption_2']]


# dropping unncessary columns
dataTrain = dataTrain.drop(dropFeatures, axis=1)
dataTest = dataTest.drop(dropFeatures, axis=1)
#let's name the categorical and numeical attributes 

#RMSLE Scorer
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#Linear Regrssion Model
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

# Train the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = dataTrain,y = yLabelsLog)


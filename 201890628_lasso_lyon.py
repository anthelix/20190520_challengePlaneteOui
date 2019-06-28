#importing the libraries
import numpy as np
import missingno as msno  # missing value
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import calendar
from vacances_scolaires_france import SchoolHolidayDates
import sys
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice



# my fonctions
def conv(data):
    data["date"] = data.timestamp.apply(lambda x : x.split('T')[0])
    data["hour"] = data.timestamp.apply(lambda x : x.split('T')[1].split(":")[0])
    data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
    data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
    data["datetime_perso"] = data.timestamp.apply(lambda x : get_format_the_date(x))
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
    
    

# IMPORTING THE DATA SET AND COMBINE
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')
dataInt.info()
#combine Input train, Input Test and  output Train
data_raw = dataInt.append(dataTest)
data = pd.merge(data_raw, dataOut, on='ID', how='left')
data.drop('ID', inplace=True, axis=1)

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
s = pd.Series(data['timestamp'])
s = pd.to_datetime(s)
data['rangeInYear'] = s.dt.strftime('%j').astype(int)
data['season'] = data['rangeInYear'].apply(lambda d : get_season(d))

## create jours working days
data['is_business_day'] = data['datetime_perso'].apply(lambda e : int(business_day(e)))

# Is it an holiday for zone A, B or C?
d = SchoolHolidayDates()
data['is_holiday'] = data['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))




## missing values

data_missing = data[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2','consumption_secondary_1', 'consumption_secondary_2','consumption_secondary_3']].copy()
sample_incomplete_rows = data[data.isnull().any(axis=1)]

# imputation par MICE
imputed_training_mice=mice(data_missing.values)
data_mice = pd.DataFrame(imputed_training_mice, columns=data_missing.columns, index = list(data.index.values))
data_mice.loc[sample_incomplete_rows.index.values]


# Imputation par KNN 
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
# start the KNN training
imputed_training_KNN=fast_knn(data_missing.values, k=30)



# Coercing To Category Type
categoryVariableList = ['weekday', 'month', 'is_business_day', 'date','hour','season','is_holiday']
for var in categoryVariableList:
    dataInt_Xy[var] = dataInt_Xy[var].astype("category")

# dropping unncessary columns
dataInt_Xy = dataInt_Xy.drop(["ID","loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3", 'timestamp', 'datetime_perso', 'rangeInYear'], axis=1)

#let's name the categorical and numeical attributes 
categorical_attributes = list(dataInt_Xy.select_dtypes(include=['object']).columns)
numerical_attributes = list(dataInt_Xy.select_dtypes(include=['float64', 'int64']).columns)
print('categorical_attributes:', categorical_attributes)
print('numerical_attributes:', numerical_attributes)

dataInt_Xy = dataInt_Xy[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2','consumption_secondary_1', 'consumption_secondary_2','consumption_secondary_3', 'date', 'hour', 'consumption_1', 'consumption_2']]


#missing value
msno.matrix(dataInt_Xy, figsize=(12,5)) # vizulaize
null_values_apptr = dataInt_Xy.isnull().sum() #count missing the same for the same locate
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr.head()

# matrice des donnees manquantes
dataInt_Xy.isnull().sum()



# imputation par MICE
imputed_training_mice=mice(data_missing.values)


# Imputation par KNN 
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
# start the KNN training
imputed_training_KNN=fast_knn(data_missing.values, k=30)

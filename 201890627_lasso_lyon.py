#importing the libraries
import numpy as np
import missingno as msno  # missing value
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import calendar
from vacances_scolaires_france import SchoolHolidayDates

# pip install git+https://github.com/novafloss/workalendar.git
# pip install vacances-scolaires-france



# my fonctions
## transform timestamp


def conv(data):
    data["date"] = data.timestamp.apply(lambda x : x.split('T')[0])
    #data['date'] =data.timestamp.apply(lambda x : x.split()[0]) #objet
    data["hour"] = data.timestamp.apply(lambda x : x.split('T')[1].split(":")[0])
    #.astype(int)
    data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
    data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
    data["datetime_perso"] = data.timestamp.apply(lambda x : get_format_the_date(x))
   
    #data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y%m%d").astype(int)
    #data["date"] = data.datetime.apply(lambda x : x.split('T')[0])
    #data['timestamp']=pd.to_datetime(data['timestamp']) 
    #data['year']=data['timestamp'].dt.year
    #data['month']=data['timestamp'].dt.month
    #data['weekday']=data['timestamp'].dt.day
    #data['hour']=data['timestamp'].dt.hour
    #data['sec']=data['timestamp'].dt.second
    #data['min']=data['timestamp'].dt.minute
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
    


# Importing the datasetda
dataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
dataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')





# Data summary

dataInt.shape

dataInt.head(2)
dataInt.dtypes
dataInt.info()
dataInt.describe()



# copy de travail
dataInt['consumption_1'] = dataOut['consumption_1']
dataInt['consumption_2'] = dataOut['consumption_2']
dataInt_Xy = dataInt.copy()


#from datetime import date
#dataInt_Xy['dateA1'] = dataInt_Xy['timestamp'].apply(lambda x : x.split('T')[0])
#dataInt_Xy['dateA2'] = dataInt_Xy['dateA1'].apply(lambda dateString : dt.datetime.strptime(dateString,"%Y-%m-%d").date())



# creating New columns from 'timestamp'
dataInt_Xy = conv(dataInt_Xy)
dataInt_Xy.dtypes
type(dataInt_Xy)
dataInt_Xy.dtypes

## create season and rangeInYear
s = pd.Series(dataInt_Xy['timestamp'])
s = pd.to_datetime(s)
dataInt_Xy['rangeInYear'] = s.dt.strftime('%j').astype(int)
#dataInt_Xy = dataInt_Xy.drop_duplicates(subset = ['rangeInYear'])                    # A ENLRVER
dataInt_Xy['season'] = dataInt_Xy['rangeInYear'].apply(lambda d : get_season(d))

## create jours working days
dataInt_Xy['is_business_day'] = dataInt_Xy['datetime_perso'].apply(lambda e : int(business_day(e)))

# Is it an holiday for zone A, B or C?
d = SchoolHolidayDates()
dataInt_Xy['is_holiday'] = dataInt_Xy['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))

dataInt_Xy.dtypes

# Coercing To Category Type
categoryVariableList = ['weekday', 'month', 'is_business_day', 'date','hour','season','is_holiday']
for var in categoryVariableList:
    dataInt_Xy[var] = dataInt_Xy[var].astype("category")

# dropping unncessary columns
dataInt_Xy = dataInt_Xy.drop(["ID","loc_1", "loc_2", "loc_secondary_1", "loc_secondary_2", "loc_secondary_3", 'timestamp', 'datetime_perso', 'rangeInYear'], axis=1)

#missing value
msno.matrix(dataInt_Xy, figsize=(12,5)) # vizulaize
null_values_apptr = dataInt_Xy.isnull().sum() #count missing the same for the same locate
null_values_apptr = null_values_apptr[null_values_apptr != 0].sort_values(ascending = False).reset_index() #only show rows with null values
null_values_apptr.columns = ["variable", "n_missing"]
null_values_apptr.head()

#let's name the categorical and numeical attributes 
categorical_attributes = list(dataInt_Xy.select_dtypes(include=['category']).columns)
numerical_attributes = list(dataInt_Xy.select_dtypes(include=['float64', 'int64']).columns)
print('categorical_attributes:', categorical_attributes)
print('numerical_attributes:', numerical_attributes)

dataInt_Xy = dataInt_Xy[['temp_1', 'temp_2', 'mean_national_temp', 'humidity_1', 'humidity_2','consumption_secondary_1', 'consumption_secondary_2','consumption_secondary_3', 'date', 'hour', 'consumption_1', 'consumption_2']]


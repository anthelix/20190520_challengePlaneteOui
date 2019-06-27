# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import calendar


# my fonctions
## transform timestamp
def conv(data):
    data["date"] = data.timestamp.apply(lambda x : x.split('T')[0])
    #data['date'] =data.timestamp.apply(lambda x : x.split()[0]) #objet
    data["hour"] = data.timestamp.apply(lambda x : x.split('T')[1].split(":")[0])
    #.astype(int)
    data["weekday"] = data.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
    data["month"] = data.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
    
   
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

## get day easter
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

## verifie si jour ferie
def is_holiday(the_date):
    """
    Vérifie si la date donnée est un jour férié
    :param the_date: datetime
    :return: bool
    """
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

## retourent jours ferie sur une periode
def business_days(date_from, date_to):
    """
    Générateur retournant les jours ouvrés dans la période [date_from:date_to]
    :param date_from: Date de début de la période
    :param date_to: Date de fin de la période
    :return: Générateur
    """
    while date_from <= date_to:
        # Un jour est ouvré s'il n'est ni férié, ni samedi, ni dimanche
        if not is_holiday(date_from) and date_from.isoweekday() not in [6, 7]:
            yield date_from
        date_from += timedelta(days=1)









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
dataInt_Xy = dataInt.copy()


#from datetime import date
dataInt_Xy['dateA1'] = dataInt_Xy['timestamp'].apply(lambda x : x.split('T')[0])
dataInt_Xy['dateA2'] = dataInt_Xy['dateA1'].apply(lambda dateString : dt.datetime.strptime(dateString,"%Y-%m-%d").date())



dataInt_Xy['is_holidays'] = dataInt_Xy['dateA2'].apply(lambda x : is_holiday(x))



# creating New columns from 'timestamp'
dataInt_Xy = conv(dataInt_Xy)
dataInt_Xy.dtypes
type(dataInt_Xy)
dataInt_Xy.dtypes
## create season and rangeInYear
s = pd.Series(dataInt_Xy['timestamp'])
s = pd.to_datetime(s)
dataInt_Xy['rangeInYear'] = s.dt.strftime('%j').astype(int)
dataInt_Xy['season'] = dataInt_Xy['rangeInYear'].apply(lambda d : get_season(d))





# create workingday 

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


# preprocess the categorical input feature



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










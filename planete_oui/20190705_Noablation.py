### changement par rapport a pdbdm
#mise sous une fonction les modofications et ajout de colonne
#agregation  dans pdbdm de dataInt et DataOUt pour avoir une meilleur moyenne sur les missing values


### import log a chercher




#importing the libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from vacances_scolaires_france import SchoolHolidayDates
import sys
from impyute.imputation.cs import mice
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Imputer
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, MultiTaskLassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#pip install vacances-scolaires-france
#conda install -c conda-forge xgboost
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
    #create jours working days
    d_tr['is_business_day'] = d_tr['datetime_perso'].apply(lambda e : int(business_day(e)))
    #Is it an holiday for zone A, B or C?
    d = SchoolHolidayDates()
    d_tr['is_holiday'] = d_tr['datetime_perso'].apply(lambda f : int(d.is_holiday(datetime.date(f))))
    d_tr['season'] = d_tr['rangeInYear'].apply(lambda d : get_season(d))
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
#------------------------------------------------------------------------------
#SPLIT
tscv = TimeSeriesSplit(n_splits=10)
print(tscv)
for train_index, test_index in tscv.split(dataInt):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataInt.iloc[train_index, :], dataInt.iloc[test_index, :]
    y_train, y_test = dataOut.iloc[train_index, :], dataOut.iloc[test_index, :]

ytrain = y_train.values.reshape(-1, 2)
ytest = y_test.values.reshape(-1,2)
ytrain.shape

#------------------------------------------------------------------------------

# jeu de donnees propres:
    # X_train et y_train pour faire le modele
    # X_test et y_test pour tester mon modele
    # dataTest pour la soumission
#--------------------------TRAIN-----------------------------------------------
Xtrain = processing(X_train)
Xtrain.head()
Xtrain.columns

#------features binary
#binary_features = ['is_holiday','is_business_day']

#------features num
num_features = ['year', 'month', 'hours']
for temp in num_features:
    Xtrain[temp] = Xtrain[temp].astype('float')
numerical_features = [f for f in Xtrain.columns if Xtrain[f].dtype == float]
scaler =  StandardScaler()
scaler.fit(Xtrain[numerical_features].values)
Xtrain[numerical_features] = scaler.transform(Xtrain[numerical_features].values)

#------features cat
categorical_features = ['season']
for var in categorical_features:
    Xtrain[var] = Xtrain[var].astype('category')
Xtrain.dtypes
list(Xtrain.columns)
Xtrain = pd.get_dummies(Xtrain, drop_first = True)
list(Xtrain.columns)
Xtrain_prep = Xtrain.values

#----------------------------TEST----------------------------------------------
Xtest = processing(X_test)
for temp in num_features:
    Xtest[temp] = Xtest[temp].astype('float')
numerical_features = [f for f in Xtest.columns if Xtest[f].dtype == float]
Xtest[numerical_features] = scaler.transform(Xtest[numerical_features].values)
for var in categorical_features:
    Xtest[var] = Xtest[var].astype('category')
Xtest.dtypes
list(Xtest.columns)
Xtest = pd.get_dummies(Xtest, drop_first = True)

# Get missing columns in the training xtest
missing_cols = set( Xtrain.columns ) - set( Xtest.columns )
# Add a missing column in Xtest set with default value equal to 0
for c in missing_cols:
    Xtest[c] = 0
# Ensure the order of column in the xtest set is in the same order than in xtrain set
Xtest = Xtest[Xtest.columns]

list(Xtest.columns)

Xtest_prep = Xtest.values

#------------------------------------------------------------------------------
print("Xtrain_prep shape: {}".format(Xtrain_prep.shape))
print("ytrain shape: {}".format(ytrain.shape))
print("Xtest_prep shape: {}".format(Xtest_prep.shape))
print("ytest shape: {}".format(ytest.shape))
#------------------------------------------------------------------------------
#-------------------------MODELING---------------------------------------------

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt



lr = LinearRegression().fit(Xtrain_prep, ytrain)
print("Training lr set score: {:.2f}".format(lr.score(Xtrain_prep, ytrain))) #0.87
print("Test lr set score: {:.2f}".format(lr.score(Xtest_prep, ytest)))   #0.81   sur apprentissge
lrn = LinearRegression()
lrn.fit(Xtrain_prep, ytrain)

ridge = Ridge().fit(Xtrain_prep, ytrain)
print("Training ridge set score: {:.2f}".format(ridge.score(Xtrain_prep, ytrain))) #0.87
print("Test ridge set score: {:.2f}".format(ridge.score(Xtest_prep, ytest))) #0.81

ridge10 = Ridge(alpha=10).fit(Xtrain_prep, ytrain)
print("Training ridge set score: {:.2f}".format(ridge10.score(Xtrain_prep, ytrain))) #0.87
print("Test ridge set score: {:.2f}".format(ridge10.score(Xtest_prep, ytest))) #0.82


rr = Ridge(alpha=0.01)
rr.fit(Xtrain_prep, y_train)
rr100 = Ridge(alpha=100)
rr100.fit(Xtrain_prep, y_train)
train_score = lr.score(Xtrain_prep, ytrain)
test_score = lr.score(Xtest_prep, ytest)
ridge_train_score = rr.score(Xtrain_prep, ytrain)
ridge_test_score = rr.score(Xtest_prep, ytest)
ridge_train_score100 = rr100.score(Xtrain_prep, ytrain)
ridge_test_score100 = rr100.score(Xtest_prep, ytest)
lasso  = Lasso()
lasso.fit(Xtrain_prep, ytrain)
train_lasso_score = lasso.score(Xtrain_prep, ytrain)
test_lasso_score = lasso.score(Xtest_prep, ytest)
coeff_used = np.sum(lasso.coef_!=0)
lasso001 = Lasso(alpha=0.01, max_iter=10e5)
lasso001.fit(Xtrain_prep, ytrain)
train_score001=lasso001.score(Xtrain_prep, ytrain)
test_score001=lasso001.score(Xtest_prep, ytest)
coeff_used001 = np.sum(lasso001.coef_!=0)
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
lasso00001.fit(Xtrain_prep, ytrain)
train_score00001=lasso00001.score(Xtrain_prep, ytrain)
test_score00001=lasso00001.score(Xtest_prep, ytest)
coeff_used00001 = np.sum(lasso00001.coef_!=0)


print ("linear regression train score:", train_score) #0.87
print ("linear regression test score:", test_score) #0.80
print ("ridge regression train score low alpha:", ridge_train_score)#0.87
print ("ridge regression test score low alpha:", ridge_test_score) #080
print ("ridge regression train score high alpha:", ridge_train_score100) # 0.87
print ("ridge regression test score high alpha:", ridge_test_score100) #0.83
print ("lasso regression train score:", train_lasso_score ) #0,86
print ("lasso regression test score:", test_lasso_score ) #0,85
print ("nombe de features:", coeff_used ) #19
print ("training score for alpha=0.01", train_score001  ) #0,85
print ("test score for alpha=0.01:", test_score001 ) #0,087
print ("nombe de features:", coeff_used ) #19
print ("training score for alpha=0.0001", train_score00001  ) #0,87
print ("test score for alpha=0.0001:", test_score00001 ) #0,80
print ("nombe de features:", coeff_used ) #19









ridge01 = Ridge(alpha=0.1).fit(Xtrain_prep, ytrain)
print("Training ridge set score: {:.2f}".format(ridge01.score(Xtrain_prep, ytrain))) #0.87
print("Test ridge set score: {:.2f}".format(ridge01.score(Xtest_prep, ytest))) #0.81



lasso1 = Lasso(fit_intercept=False,normalize=False).fit(Xtrain_prep, ytrain)
print(lasso1)
print("Training Lasso set score: {:.2f}".format(lasso1.score(Xtrain_prep, ytrain))) #0.54
print("Test lasso set score: {:.2f}".format(lasso01.score(Xtest_prep, ytest)))   #-6.19
from sklearn.linear_model import lasso_path
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.metrics import r2_score

reg = MultiTaskLassoCV(cv=5, random_state=0).fit(Xtrain_prep, ytrain)
print(r2_score(ytrain, reg.predict(Xtrain_prep))) #0.8
print(reg.alpha_) #0.6


my_alphas = np.array([0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0])

lcv = MultiTaskLassoCV(alphas=my_alphas,random_state=0,cv=5)
lcv.fit(Xtrain_prep,ytrain)
print(r2_score(ytrain,lcv.predict(Xtrain_prep))) #0.87
print(lcv.alpha_)
print(lcv.mse_path_)#0.8
avg_mse = np.mean(lcv.mse_path_, axis=1)
print(pd.DataFrame({'alpha': lcv.alphas_, 'MSE':avg_mse}))


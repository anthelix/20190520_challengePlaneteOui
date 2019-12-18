#importing the libraries
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from datetime import datetime, timedelta
from vacances_scolaires_france import SchoolHolidayDates
#import sys
#from impyute.imputation.cs import mice
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
#from fancyimpute import KNN
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
#pip install missingpy
#from missingpy import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error

#pip install vacances-scolaires-france
#pip install impyute
#pip3 install xgboost
#pip install fancyimpute

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
     
def processing(dataInt):
    ## missing value
    df = dataInt.copy()
    df_num = df.drop(['timestamp','loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
    df_NonNum = df.select_dtypes(include=[np.object])
    #train_cols = list(df_num)
    #df_clean = pd.DataFrame(KNN(k=5).complete(df_num))
    #df_clean.columns = train_cols
    #knn_impute(target=df_num['temp_1', 'temp_2', 'mean_national_temp'], attributes=df_num.drop(['humidity_1', 'humidity_2'], 1),
               #aggregation_method='median', k_neighbors=40, numeric_distance='euclidean')




    #imputed_training_mice = mice(df_num.values)
    #data_mice = pd.DataFrame(imputed_training_mice, columns = df_num.columns, index = list(df.index.values))
    #dClean = data_mice.join(df_NonNum)


    ## drop variable inutile
    d_tr = df.drop(['loc_1', 'loc_2', 'loc_secondary_1', 'loc_secondary_2', 'loc_secondary_3'], axis=1)
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

#metric donner par l"ENS
def weighted_mean_absolute_error(dataframe_1, dataframe_2):
    """Weighted mean absolute error regression loss

        ----------
        y_true : array-like of shape = (n_samples,2)
        Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples,2)
        Estimated target values.

        """

    y_true = dataframe_1.values
    y_pred = dataframe_2.values
    c12 = np.array([1136987, 1364719])

    return 2 * metrics.mean_absolute_error(y_true*c12, y_pred*c12) / np.sum(c12)


#------------------------------------------------------------------------------    

# creere un je de test
dataInt_raw = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
dataOut_raw = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')
data_blink = pd.concat([dataInt_raw, dataOut_raw[['consumption_1', 'consumption_2']]], axis=1)

#----------------------
dataInt = dataInt_raw.drop(['ID'], axis=1)
dataOut = dataOut_raw.drop(['ID'], axis=1)

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
ytest = y_test.values.reshape(-1, 2)
ytrain.shape

#------------------------------------------------------------------------------
#--------------------------TRAIN-----------------------------------------------
Xtrain1 = processing(X_train)
Xtest1 = processing(X_test)
Xtrain1.head()
Xtrain1.columns
Xtrain = Xtrain1.copy()
Xtest = Xtest1.copy()
#------------------------------------------------------------------------------
print("Xtrain shape: {}".format(Xtrain.shape))
print("ytrain shape: {}".format(ytrain.shape))
print("Xtest shape: {}".format(Xtest.shape))
print("ytest shape: {}".format(ytest.shape))
Xtrain.dtypes


#------features binary
#binary_features = ['is_holiday','is_business_day']


#------features num
#--train
#num_features = ['year', 'month', 'hours']
#for temp in num_features:
#    Xtrain[temp] = Xtrain[temp].astype('float')

consum_varaiables = ['consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3' ]
Xtrain['consum_mean'] = Xtrain[consum_varaiables].mean(axis=1)
Xtest['consum_mean'] = Xtest[consum_varaiables].mean(axis=1)
Xtrain = Xtrain.drop(['consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3'], axis=1)
Xtest= Xtest.drop(['consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3'], axis=1)

numerical_features = [f for f in Xtrain.columns if Xtrain[f].dtype == float]
scaler =  StandardScaler()
scaler.fit(Xtrain[numerical_features].values)
Xtrain[numerical_features] = scaler.transform(Xtrain[numerical_features].values)
#--test
numerical_features = [f for f in Xtest.columns if Xtest[f].dtype == float]
Xtest[numerical_features] = scaler.transform(Xtest[numerical_features].values)
#------features cat
categorical_features = ['year', 'month', 'hours', 'season', 'is_holiday','is_business_day']
for var in categorical_features:
    Xtrain[var] = Xtrain[var].astype('category')
    Xtest[var] = Xtest[var].astype('category')


all_data = pd.concat([Xtrain, Xtest])
for column in categorical_features:
    Xtrain[column] = Xtrain[column].astype(CategoricalDtype(categories = all_data[column].unique(), ordered=True))
    Xtest[column] = Xtest[column].astype(CategoricalDtype(categories = all_data[column].unique(), ordered=True))

Xtrain = pd.get_dummies(Xtrain, drop_first=True)
Xtest = pd.get_dummies(Xtest, drop_first=True)
Xtrain = Xtrain[Xtrain.columns]
Xtest = Xtest[Xtest.columns]
list(Xtrain.columns)
list(Xtest.columns)
Xtrain_prep = Xtrain.values
Xtest_prep = Xtest.values

#------------------------------------------------------------------------------
#-------------------------MODELING---------------------------------------------

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

classifiers = [['linear Regression :' ,LinearRegression()],
                ['Ridge01 :' ,Ridge(alpha=0.01)],
                ['Ridge100 :' ,Ridge(alpha=100)],
                ['Lasso :' , Lasso()],
                ['Lasso alpha.1 :' , Lasso(alpha=0.1, max_iter=10e5)],
                ['Lasso1 alpha.01 :', Lasso(alpha=0.01, max_iter=10e5)],
                ['Lasso alpha00001 :', Lasso(alpha=0.0001, max_iter=10e5)],
                ['Elastic Net ratio 0,5 :', ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)],
                ['Elasic Net ratio 1:',  ElasticNet(alpha=1, l1_ratio=1, normalize=False)],
                ['Multi Output GBR :', MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1)],
                ['Multi Output RFR :', MultiOutputRegressor(RandomForestRegressor(min_samples_split=5, 
                                                                                  n_estimators=70, 
                                                                                  max_depth=40, 
                                                                                  random_state=0,
                                                                                  min_samples_leaf=2), n_jobs=-1)]]

print("Accuracy Results ...")
from sklearn.metrics import r2_score
#   RMSE erreur quadratique moyenne(erreur d'estimation)
#   R_sqaure qualite moyenne de regression

for name, classifier in classifiers:
    classifier = classifier
    classifier.fit(Xtrain_prep, ytrain)
    predictions = classifier.predict(Xtest_prep)
    #print("{} train_score {:.2f}".format(name, classifier.score(Xtrain_prep, ytrain)))
    #print('le score y R2 est {:.2f}'.format(r2_score(ytrain, classifier.predict(Xtrain_prep))))
    #print("{} test_score {:.2f}".format(name, classifier.score(Xtest_prep, ytest)))
    #print('le score y R2 est {:.2f}'.format(r2_score(ytest, classifier.predict(Xtest_prep))))
    #print("{}  RMSE {:.2f}".format(name, np.sqrt(metrics.mean_squared_error(ytest, predictions))))
    #print("{}   MSE {:.2f}".format(name, mean_squared_error(ytest, predictions)))
    print("{}   MAE {:.2f}".format(name, metrics.mean_absolute_error(ytest, predictions)))
    print("Metric de l'ENS {:.2f}".format(weighted_mean_absolute_error(pd.DataFrame(ytest), pd.DataFrame(predictions))))
    #print(10*' ' + '>>' + 5*'-' + '<<')

y_train.head(5)
#-------------------------submiting---------------------------------------------
Xtest = Xtest[Xtest.columns]
list(Xtrain.columns)
list(Xtest.columns)
Xtrain_prep = Xtrain.values
Xtest_prep = Xtest.values
my_model =MultiOutputRegressor(RandomForestRegressor(min_samples_split=5,
                                                     n_estimators=70, 
                                                     max_depth=40, 
                                                     random_state=0,
                                                     min_samples_leaf=2), n_jobs=-1)
my_model.fit(Xtrain_prep, ytrain)


#-------------------
datatest_raw = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
datatest = datatest_raw.drop(['ID'], axis=1)
datatest1= processing(datatest)
consum_varaiables = ['consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3' ]
datatest1['consum_mean'] = datatest1[consum_varaiables].mean(axis=1)
datatest= datatest1.drop(['consumption_secondary_1', 'consumption_secondary_2', 'consumption_secondary_3'], axis=1)
numerical_features = [f for f in datatest.columns if datatest[f].dtype == float]
datatest[numerical_features] = scaler.transform(datatest[numerical_features].values)
for var in categorical_features:
    datatest[var] = datatest[var].astype('category')
for column in categorical_features:
    datatest[column] = datatest[column].astype(CategoricalDtype(categories = all_data[column].unique(), ordered=True))
datatest = pd.get_dummies(datatest, drop_first=True)
datatest = datatest[datatest.columns]
datatest_val = datatest.values
predicted_data = my_model.predict(datatest_val)
print(predicted_data[:,0]) 
#my_submission3 = pd.DataFrame({'ID': datatest_raw.ID, 'consumption_1': predicted_data[:,0], 'consumption_2': predicted_data[:,1] })
#my_submission3.to_csv('submission3.csv', index=False)


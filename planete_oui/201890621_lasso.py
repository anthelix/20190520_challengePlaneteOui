
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# Importing the dataset
ElecdataInt = pd.read_csv('./data_set1/input_training_ssnsrY0.csv')
ElecdataTest = pd.read_csv('./data_set1/input_test_cdKcI0e.csv')
ElecdataOut = pd.read_csv('./data_set1/output_training_Uf11I9I.csv')

import datetime as dt
from datetime import datetime
import calendar

print(ElecdataInt.describe())
ElecdataInt.dtypes
ElecdataInt.info

ElecdataInt['consumption_1'] = ElecdataOut['consumption_1']
ElecdataInt['consumption_2'] = ElecdataOut['consumption_2']

ElecdataInt.dtypes

ElecdataInt["date"] = ElecdataInt.timestamp.apply(lambda x : x.split('T')[0])
ElecdataInt["hour"] = ElecdataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
ElecdataInt["date"] = pd.to_datetime(ElecdataInt['date'], format="%Y/%m/%d")

ElecdataInt["loc_1"] = ElecdataInt.loc_1.apply(lambda x : x[1:-1])
ElecdataInt["loc_1_latitude"] = ElecdataInt.loc_1.apply(lambda x : x.split(',')[0])
ElecdataInt["loc_1_longitude"] = ElecdataInt.loc_1.apply(lambda x : x.split(',')[1])

ElecdataInt["loc_2"] = ElecdataInt.loc_2.apply(lambda x : x[1:-1])
ElecdataInt["loc_2_latitude"] = ElecdataInt.loc_2.apply(lambda x : x.split(',')[0])
ElecdataInt["loc_2_longitude"] = ElecdataInt.loc_2.apply(lambda x : x.split(',')[1])

ElecdataInt["loc_secondary_1"] = ElecdataInt.loc_secondary_1.apply(lambda x : x[1:-1])
ElecdataInt["loc_secondary_1_latitude"] = ElecdataInt.loc_secondary_1.apply(lambda x : x.split(',')[0])
ElecdataInt["loc_secondary_1_longitude"] = ElecdataInt.loc_secondary_1.apply(lambda x : x.split(',')[1])


ElecdataInt["loc_secondary_2"] = ElecdataInt.loc_secondary_2.apply(lambda x : x[1:-1])
ElecdataInt["loc_secondary_2_latitude"] = ElecdataInt.loc_secondary_2.apply(lambda x : x.split(',')[0])
ElecdataInt["loc_secondary_2_longitude"] = ElecdataInt.loc_secondary_2.apply(lambda x : x.split(',')[1])

ElecdataInt["loc_secondary_3"] = ElecdataInt.loc_secondary_3.apply(lambda x : x[1:-1])
ElecdataInt["loc_secondary_3_latitude"] = ElecdataInt.loc_secondary_3.apply(lambda x : x.split(',')[0])
ElecdataInt["loc_secondary_3_longitude"] = ElecdataInt.loc_secondary_3.apply(lambda x : x.split(',')[1])

ElecdataInt = ElecdataInt.drop(["timestamp", "ID", "loc_1", "loc_2","loc_secondary_1", "loc_secondary_2", "loc_secondary_3" ],axis=1)





#subdivision
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(ElecdataInt, test_size = 0.2)

from sklearn.model_selection import train_test_split
y_train, y_test = train_test_split(ElecdataOut, test_size = 0.2)


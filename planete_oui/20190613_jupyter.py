import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import os # Le module d'OS en Python fournit un moyen d'utiliser les fonctionnalités dépendantes du système d'exploitation. 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
import warnings
import matplotlib as mpl
import datetime as dt
from datetime import datetime
import calendar

sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')

rawData_trainInput = pd.read_csv('data_set1/input_training_ssnsrY0.csv')
rawData_trainOutput = pd.read_csv('data_set1/output_training_Uf11I9I.csv')
rawData_testInput = pd.read_csv('data_set1/input_test_cdKcI0e.csv')
trainInput = rawData_trainInput.copy(deep=True)
trainOutput = rawData_trainOutput.copy(deep=True)
testInput = rawData_testInput.copy(deep=True)

print("TrainInput:\n", trainInput.isnull().sum())
print('-'*10)
print("TestInput:\n", testInput.isnull().sum())    
print('-'*10)
print("TrainOutput:\n", trainOutput.isnull().sum())

trainInput.info()
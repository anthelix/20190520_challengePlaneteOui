
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

print(dataInt.describe())

dataInt["date"] = dataInt.timestamp.apply(lambda x : x.split('T')[0])
dataInt["hour"] = dataInt.timestamp.apply(lambda x : x.split('T')[1].split(":")[0]).astype(int)
dataInt["date"] = pd.to_datetime(dataInt['date'], format="%Y/%m/%d")

dataInt = dataInt.drop(["timestamp", "ID"],axis=1)

import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

groupBy_whole = dataInt.groupby('date').mean()

trace1 = go.Scatter(
        x=groupBy_whole.temp_1.index, 
        y=groupBy_whole.temp_1, 
        mode = 'lines',
        name = 'temp_1',
        marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
trace2 = go.Scatter(
        x=groupBy_whole.temp_2.index, 
        y=groupBy_whole.temp_2, 
        mode = 'lines',
        name = 'temp_2',
        marker = dict(color = 'rgba(16, 112, 1, 0.8)'))

data = [trace1, trace2]

layout=dict(title="Time Series Plot for Mean Daily temperature", xaxis={'title':'Date'}, yaxis={'title':'Temp'})
fig=dict(data=data,layout=layout)
iplot(fig)


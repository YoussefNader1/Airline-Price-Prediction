from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *

#Load players data
data = pd.read_csv('airline-price-prediction.csv')

#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)

#Features
X=data.iloc[:,0:10] #Features
#OutPut
Y=data['price']


#PreProcessing
#Stop Column
X['stop'] = (X["stop"].str.split("-", n = 1, expand = True))[0]
X['stop']=X['stop'].replace('non','0')
X['stop']=X['stop'].replace('2+','2')
X['stop']=pd.to_numeric(X['stop'])
#dep_time Column
X['dep_time']=X['dep_time'].str.replace(':','.')
X['dep_time']=pd.to_numeric(X['dep_time'])
#arr_time Column
X['arr_time']=X['arr_time'].str.replace(':','.')
X['arr_time']=pd.to_numeric(X['arr_time'])
#route Column
new=X["route"].str.split(", ")
source = []
destination = []
for x in new:
    s = str(x[0]).split(": ")
    source.append(s[1])
    d = str(x[1]).split(": ")
    d = str(d[1]).split("}")
    destination.append(d[0])
X = X.drop(['route'], axis = 1)
X['source'] = source
X['destination']=destination

#timeTaken Column
X['time_taken']=X['time_taken'].str.replace('h m','')
X['time_taken']=X['time_taken'].str.replace('h ','.')
X['time_taken']=X['time_taken'].str.replace('m','')
X['time_taken']=pd.to_numeric(X['time_taken'])

#Date Column
newDate = X['date']
columnDate = []
day=[]
month=[]
for i in range(len(data)):
    if newDate[i][-5] == "-":
        columnDate.append(datetime.strptime(newDate[i], '%d-%m-%Y'))
    elif newDate[i][-5] == "/":
        columnDate.append(datetime.strptime(newDate[i], '%d/%m/%Y'))
    day.append(columnDate[i].day)
    month.append(columnDate[i].month)

X = X.drop(['date'], axis = 1)
X['day']=day
X['month']=month

X['day']=pd.to_numeric(X['day'])
X['month']=pd.to_numeric(X['month'])

#Encoding
cols=('airline','ch_code','type','source','destination')
X=Feature_Encoder(X,cols)

#Feature Scaling
X=featureScaling(X,0,10) #Look At Meeeee Agaaaaiiiiinnnnnnnnn

print(X.head())


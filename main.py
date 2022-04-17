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
X['stop'] = (X["stop"].str.split("-", n = 1, expand = True))[0]
X['stop']=X['stop'].replace('non','0')
X['stop']=X['stop'].replace('2+','2')


#Encoding
cols=('airline','arr_time','ch_code','type','dep_time')
X=Feature_Encoder(X,cols)


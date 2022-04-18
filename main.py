import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from Correlation import *

#Load players data
data = pd.read_csv('airline-price-prediction.csv')
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
#Features
X = data.iloc[:,0:10]
#OutPut
Y = data['price']

#Calling pre-processing function to process and reformate the data
X = X_preprocessData(X, data)

#Encoding
cols = ('airline', 'ch_code', 'type', 'source', 'destination')
X = Feature_Encoder(X, cols)

#Feature Scaling
X = featureScaling(X, 0, 10) #Look At Meeeee Agaaaaiiiiinnnnnnnnn

#Price processing
Y = Y_preprocessData(Y)

top_features = correlation(X,Y)

#singlevar has type columne whice has the highest correlation
singleVar = top_features[0]

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
X = featureScaling(X, 0, 1) #Look At Meeeee Agaaaaiiiiinnnnnnnnn
#Price processing
Y = Y_preprocessData(Y)

top_features = correlation(X, Y)

X = X[top_features]
# X = X.replace([np.inf, -np.inf], np.nan)
# X = X.fillna(method='bfill', inplace=True)
# X = X.dropna(inplace=True)
# X = X.dropna(how='any',inplace=True)
# x = X.to_numpy()
# print(type(x))
# print(x)
# X = X.iloc[:,:].values
# X = X.reshape(-1,1)

cls = linear_model.LinearRegression()
cls.fit(X,Y)
Prediction = cls.predict(X)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), Prediction))

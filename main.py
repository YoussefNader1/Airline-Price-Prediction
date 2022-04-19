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
#data.dropna(how='any',inplace=True)
#Features
X = data.iloc[:,0:10]
#OutPut
Y = data['price']

#Calling pre-processing function to process and reformate the data
X = X_preprocessData(X, data)


# X = pd.get_dummies(X)
# print(X.shape)
# print(X)
#Encoding
cols = ('airline', 'ch_code', 'type', 'source', 'destination')
X = Feature_Encoder(X, cols)

#Feature Scaling
X = featureScaling(X, 0, 10) #Look At Meeeee Agaaaaiiiiinnnnnnnnn
#Price processing
Y = Y_preprocessData(Y)

top_features = correlation(X, Y)

X = X[top_features]
# DataFrame to Arraaaaaaaaayy
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.20,shuffle=True) #Features splitting

#change dataframes into arrays
X=np.array(X)
Y=np.array(Y)

#Drop nulls for train data
x_train.dropna(how='any',inplace=True)
y_train.dropna(how='any',inplace=True)
x_test.fillna(0)
y_test.fillna(0)


print("Enter 1 for multiple regression model\nEnter 2 for polynomial regression")
choice = int(input("Choose your model: "))
if choice == 1:

    cls = linear_model.LinearRegression()
    cls.fit(x_train,y_train)
    Prediction = cls.predict(x_test)

    print(cls.score(x_test, y_test))
    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), Prediction))

    true_price_value = np.asarray(y_test)[0]
    predicted_price_value = Prediction[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

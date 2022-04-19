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

    # plt.style.use('default')
    # plt.style.use('ggplot')
    #
    # fig, ax = plt.subplots(figsize=(7, 3.5))
    #
    # ax.plot(x_train, y_train, color='k', label='Regression model')
    # ax.scatter(x_train, Prediction, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
    # ax.set_ylabel('Gas production (Mcf/day)', fontsize=14)
    # ax.set_xlabel('Porosity (%)', fontsize=14)
    # ax.legend(facecolor='white', fontsize=11)
    # ax.text(0.55, 0.15, '$y = %.2f x_1 - %.2f $' % (cls.coef_[0], abs(cls.intercept_)), fontsize=17,
    #         transform=ax.transAxes)
    #
    # fig.tight_layout()

    print(cls.score(x_test, y_test))
    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), Prediction))

    true_price_value = np.asarray(y_test)[0]
    predicted_price_value = Prediction[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

elif choice == 2:
    poly_features = PolynomialFeatures(degree=9)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    ypred = poly_model.predict(poly_features.transform(x_test))

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(x_test))

    # print(poly_model.score(x_test, y_test))

    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

    true_player_value = np.asarray(y_test)[0]
    predicted_player_value = prediction[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))

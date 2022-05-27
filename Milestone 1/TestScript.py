import pickle
# import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from Correlation import *
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import time

# Load Airline data
# data = pd.read_csv('airline-price-prediction.csv')

data = pd.read_csv('airline-test-samples.csv')

# Features
X = data.iloc[:, 0:10]
# Label
Y = data['price']

# Feature processing
X = X_preprocessData(X, data)
# Label processing
Y = Y_preprocessData(Y)

# Feature selection
X = pd.DataFrame(X, columns=['ch_code', 'num_code', 'time_taken', 'stop', 'type'])

# Feature encoding
X['type'] = Feature_Encoder_Type(X)
type = X['type']
X['ch_code'] = Feature_Encoder_ch_code(X)

# x_train, X, y_train, Y = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=1)

# Feature Scaling
X = featureScalingTestScript(X, 0, 1)

# change dataframes into arrays
X = np.array(X)
Y = np.array(Y)

print("Enter 1 for multiple regression model\nEnter 2 for polynomial regression")
choice = int(input("Choose your model: "))

if choice == 1:
    cls = pickle.load(open('linear_regression.pkl', 'rb'))
    Prediction = cls.predict(X)

    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('R2 Score', metrics.r2_score(Y, Prediction))
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), Prediction))

    true_price_value = np.asarray(Y)[0]
    predicted_price_value = Prediction[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

#
# elif choice == 2:
#     #poly_features = PolynomialFeatures(degree=5)
#
#     poly_model = pickle.load(open('poly_features.pkl', 'rb'))
#     joblib.dump(poly_model, 'model_jlib')
#     m_jlib = joblib.load('model_jlib')
#     m_jlib.predict(X)
#     # transforms the existing features to higher degree features.
#     #X_train_poly = poly_features.fit_transform(x_train)
#     # fit the transformed features to Linear Regression
#     #poly_model = linear_model.LinearRegression()
#     #startTrain = time.time()
#     #poly_model.fit(X_train_poly, y_train)
#     #endTrain = time.time()
#     # predicting on training data-set
#     #y_train_predicted = poly_model.predict(X_train_poly)
#     #start_test = time.time()
#     #prediction = poly_model.predict(X)
#     #end_test = time.time()
#     # predicting on test data-set
#     #prediction = poly_model.predict(poly_features.fit_transform(X))
#     #print('Co-efficient of linear regression', poly_model.coef_)
#     #print('Intercept of linear regression model', poly_model.intercept_)
#     #print('R2 Score', metrics.r2_score(Y, prediction))
#     #print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
#     #print("Actual time for training", endTrain - startTrain)
#     #print("Actual time for Testing", end_test - start_test)
#
#     #true_price_value = np.asarray(Y)[0]
#     #predicted_price_value = prediction[0]
#
#     #print("The true price value " + str(true_price_value))
#     #print("The predicted price value " + str(predicted_price_value))

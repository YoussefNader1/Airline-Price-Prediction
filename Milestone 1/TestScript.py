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
import joblib

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

# x_train, X, y_train, Y = train_test_split(X, Y, test_size=1, shuffle=True, random_state=1)

# Feature Scaling
X = featureScalingTestScript(X, 0, 1)

X = X.fillna(0)
Y = Y.fillna(0)
# change dataframes into arrays
X = np.array(X)
Y = np.array(Y)

print("Enter 1 for multiple regression model\nEnter 2 for polynomial regression")
choice = int(input("Choose your model: "))


def model():
    print("Linn")
    cls = pickle.load(open('linear_regression.pkl', 'rb'))
    Prediction1 = cls.predict(X)

    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('R2 Score', metrics.r2_score(Y, Prediction1))
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y), Prediction1))
    true_price_value = np.asarray(Y)[0]
    predicted_price_value = Prediction1[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

    print("PP")
    print(Prediction1)
    print("YYYYYY")
    print(Y)

    print("POLYYYYY")
    poilynomia_features_model = joblib.load('poilynomia_features_model')
    poly_model = joblib.load('polymodel')

    X_val_prep = poilynomia_features_model.transform(X)
    prediction = poly_model.predict(X_val_prep)

    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('R2 Score', metrics.r2_score(Y, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    true_price_value = np.asarray(Y)[0]
    predicted_price_value = prediction[0]
    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

    print("PP")
    print(prediction)
    print("YYYYYY")
    print(Y)


if choice == 0:
    model()
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




elif choice == 2:
    poilynomia_features_model = joblib.load('poilynomia_features_model')
    poly_model = joblib.load('polymodel')

    X_val_prep = poilynomia_features_model.transform(X)
    prediction = poly_model.predict(X_val_prep)

    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('R2 Score', metrics.r2_score(Y, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    true_price_value = np.asarray(Y)[0]
    predicted_price_value = prediction[0]
    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

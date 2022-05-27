import joblib
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from Correlation import *
import time
import pickle

# Load Airline data
data = pd.read_csv('airline-price-prediction.csv')
# Features
X = data.iloc[:, 0:10]
# Label
Y = data['price']

# Feature processing
X = X_preprocessData(X, data)
# Label processing
Y = Y_preprocessData(Y)

# Feature encoding
cols = ('airline', 'ch_code', 'type', 'source', 'destination')  # column to be encoded
X = Feature_Encoder(X, cols)

# Feature Scaling
X = featureScaling(X, 0, 1)

# Feature selection
top_features = correlation(X, Y)
X = X[top_features]
# Data distribution
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,
                                                    random_state=1)  # Features splitting

# change dataframes into arrays
X = np.array(X)
Y = np.array(Y)

# Drop nulls for train data
x_train.dropna(how='any', inplace=True)
y_train.dropna(how='any', inplace=True)
# Replace null for test data
x_test.fillna(0,inplace=True)
y_test.fillna(0,inplace=True)

print("Enter 1 for multiple regression model\nEnter 2 for polynomial regression")
choice = int(input("Choose your model: "))

if choice == 1:
    # cls = linear_model.LinearRegression()
    cls = pickle.load(open('linear_regression.pkl', 'rb'))
    # startTrain = time.time()
    # cls.fit(x_train, y_train)
    # endTrain = time.time()
    # start_test = time.time()
    Prediction = cls.predict(x_test)
    # end_test = time.time()

    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('R2 Score', metrics.r2_score(y_test, Prediction))
    print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), Prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

    true_price_value = np.asarray(y_test)[0]
    predicted_price_value = Prediction[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))


elif choice == 2:
    poly_features = PolynomialFeatures(degree=5)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(x_train)
    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    startTrain = time.time()
    poly_model.fit(X_train_poly, y_train)
    endTrain = time.time()
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    start_test = time.time()
    ypred = poly_model.predict(poly_features.transform(x_test))
    end_test = time.time()
    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(x_test))
    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    print("Actual time for training", endTrain - startTrain)
    print("Actual time for Testing", end_test - start_test)

    true_price_value = np.asarray(y_test)[0]
    predicted_price_value = prediction[0]

    print("The true price value " + str(true_price_value))
    print("The predicted price value " + str(predicted_price_value))

    #joblib.dump(poly_model, 'polymodel')
    #joblib.dump(poly_features, 'poilynomia_features_model')

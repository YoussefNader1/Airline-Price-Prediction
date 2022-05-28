import pickle
# import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Pre_processing import *
from Correlation import *
from sklearn import svm
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import time

data = pd.read_csv('airline-test-samples.csv')

# Features
X = data.iloc[:, 0:10]
# Label
Y = data.iloc[:, -1]

Y = pd.DataFrame(Y)

# Feature processing
X = X_preprocessData(X, data)

# Feature selection
X = pd.DataFrame(X, columns=['ch_code', 'num_code', 'time_taken', 'stop', 'type', 'month'])

# print(X)


# Feature encoding
X['type'] = Feature_Encoder_Type(X)
type = X['type']
X['ch_code'] = Feature_Encoder_ch_code(X)
Y = Feature_Encoder_TicketCategory(Y)

# Feature Scaling
X = featureScalingTestScript(X, 0, 1)

g = pd.Series(Y, name='TicketCategory')
Y = pd.DataFrame(g)

X = X.fillna(0)
Y = Y.fillna(0)

X = np.array(X)
Y = np.array(Y)

C = 0.001  # SVM regularization parameter


def model():
    print("lin")
    pickled_model_linear = pickle.load(open('lin.pkl', 'rb'))
    prediction = pickled_model_linear.predict(X)
    print("Accuracy linear:", metrics.accuracy_score(Y, prediction))
    print('R2 Score', metrics.r2_score(Y, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))

    print("poly")
    pickled_model_linear1 = pickle.load(open('poly.pkl', 'rb'))
    prediction2 = pickled_model_linear1.predict(X)
    print("Accuracy linear:", metrics.accuracy_score(Y, prediction2))
    print('R2 Score', metrics.r2_score(Y, prediction2))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction2))

    print("rbf")
    pickled_model_linear2 = pickle.load(open('rbf.pkl', 'rb'))
    prediction2 = pickled_model_linear2.predict(X)
    print("Accuracy linear:", metrics.accuracy_score(Y, prediction2))
    print('R2 Score', metrics.r2_score(Y, prediction2))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction2))

    print("kernel")
    pickled_model_linear3 = pickle.load(open('kernel.pkl', 'rb'))
    prediction3 = pickled_model_linear3.predict(X)
    print("Accuracy linear:", metrics.accuracy_score(Y, prediction3))
    print('R2 Score', metrics.r2_score(Y, prediction3))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction3))

    print("LogisticRegression")
    pickled_model_linear4 = pickle.load(open('LogisticRegression.pkl', 'rb'))
    prediction4 = pickled_model_linear4.predict(X)
    print("Accuracy linear:", metrics.accuracy_score(Y, prediction4))
    print('R2 Score', metrics.r2_score(Y, prediction4))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction4))

    print("DT")
    # loading Decision Tree model
    pickled_model_DecisionTree5 = pickle.load(open('DT.pkl', 'rb'))
    prediction5 = pickled_model_DecisionTree5.predict(X)
    print('R2 Score', metrics.r2_score(Y, prediction5))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction5))


model()

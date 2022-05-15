import pickle

import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from Correlation import *
from sklearn import svm, datasets
#
# import time

# Load Airline data
data = pd.read_csv('airline-price-classification.csv')
# Features
X = data.iloc[:, 0:10]
# Label
Y = data.iloc[:, -1]

# Y.to_frame(name='my_column_name')
Y = pd.DataFrame(Y)

# Feature processing
X = X_preprocessData(X, data)
# Label processing
# Y = Y_preprocessData(Y)

# Feature encoding
cols = ('airline', 'ch_code', 'type', 'source', 'destination')  # column to be encoded
X = Feature_Encoder(X, cols)

# Feature Scaling
X = featureScaling(X, 0, 1)

print(Y)
print(type(Y))
Y = Feature_Encoder_TicketCategory(Y)

print(Y)
print(type(Y))
g = pd.Series(Y, name='TicketCategory')
Y = pd.DataFrame(g)
print(Y)

# Feature selection
top_features = correlation(X, Y)
X = X[top_features]

print(X)

# Data distribution
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,
                                                    random_state=1)  # Features splitting
pickle.dump(y_test, open('y_test.pkl', 'wb'))
pickle.dump(x_test, open('x_test.pkl', 'wb'))

# change dataframes into arrays
X = np.array(X)
Y = np.array(Y)

# Drop nulls for train data
x_train.dropna(how='any', inplace=True)
y_train.dropna(how='any', inplace=True)
# Replace null for test data
x_test.fillna(0)
y_test.fillna(0)

# we create an instance of SVM and fit out data.
C = 0.001  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train.values.ravel())
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(x_train, y_train.values.ravel())
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train.values.ravel())
pickle.dump(lin_svc, open('lin.pkl', 'wb'))

pickled_model = pickle.load(open('lin.pkl', 'rb'))
xtestload = pickle.load(open('x_test.pkl', 'rb'))
ytestload = pickle.load(open('y_test.pkl', 'rb'))

k=pickled_model.predict(xtestload)


# print("Accuracy linear:", metrics.accuracy_score(y_test, lin_svc.predict(x_test)))
print("Accuracy linear:", metrics.accuracy_score(ytestload, k))
print('R2 Score', metrics.r2_score(ytestload, k))
print('Mean Square Error', metrics.mean_squared_error(ytestload, k))
# print("Accuracy rbf:", metrics.accuracy_score(y_test, rbf_svc.predict(x_test)))
# print("Accuracy poly:", metrics.accuracy_score(y_test, poly_svc.predict(x_test)))
# print(predictions)


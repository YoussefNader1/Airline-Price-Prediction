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
cols = ('airline', 'source', 'destination', 'ch_code')  # column to be encoded
X = Feature_Encoder(X, cols)
X['type'] = Feature_Encoder_Type(X)
# Feature Scaling
X = featureScaling(X, 0, 1)

Y = Feature_Encoder_TicketCategory(Y)

g = pd.Series(Y, name='TicketCategory')
Y = pd.DataFrame(g)

# Feature selection
top_features = correlation(X, Y)
X = X[top_features]

print(X)

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

C = 0.001  # SVM regularization parameter
print("Choose your model: \n")
print("1 - Linear svm model")
print("2 - Polynomial svm model")
print("3 - RBF svm model")
print("4 - Linear kernel svm model")
print("5 - Logistic Regression model")
print('6 - Decision tree model')
choice = int(input("Enter your choice: "))

if choice == 1:

    # startTrain = time.time()
    # lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving linear model
    # pickle.dump(lin_svc, open('lin.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading linear model
    pickled_model_linear = pickle.load(open('lin.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------

    # start_test = time.time()
    prediction = pickled_model_linear.predict(x_test)
    # end_test = time.time()

    print("Accuracy linear:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)


elif choice == 2:

    # startTrain = time.time()
    # poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving polynomial model
    # pickle.dump(poly_svc, open('poly.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading polynomial model
    pickled_model_polynomial = pickle.load(open('poly.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_polynomial.predict(x_test)
    # end_test = time.time()

    print("Accuracy polynomial:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

# RBF Model
elif choice == 3:

    # startTrain = time.time()
    # rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving RBF model
    # pickle.dump(rbf_svc, open('rbf.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading rbf model
    pickled_model_rbf = pickle.load(open('rbf.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_rbf.predict(x_test)
    # end_test = time.time()

    print("Accuracy rbf:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)


# Kernel Model
elif choice == 4:

    # startTrain = time.time()
    # svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving Linear kernel model
    # pickle.dump(svc, open('kernel.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading Kernel model
    pickled_model_kernel = pickle.load(open('kernel.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_kernel.predict(x_test)
    # end_test = time.time()

    print("Accuracy kernel:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

# Logistic Regression Model
elif choice == 5:

    # startTrain = time.time()
    # LRG = linear_model.LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving Logistic Regression model
    # pickle.dump(LRG, open('LogisticRegression.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading Logistic Regression model
    pickled_model_LogisticRegression = pickle.load(open('LogisticRegression.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_LogisticRegression.predict(x_test)
    # end_test = time.time()

    print("Accuracy Logistic Regression:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

# Decision Tree Model
elif choice == 6:

    # startTrain = time.time()
    # DT = DecisionTreeRegressor(random_state=0, max_depth=3, min_samples_leaf=5).fit(x_train, y_train.values.ravel())
    # endTrain = time.time()

    # saving Decision Tree model
    # pickle.dump(DT, open('DT.pkl', 'wb'))
    # ----------------------------------------------------------------------------------------------------
    # loading Decision Tree model
    pickled_model_DecisionTree = pickle.load(open('DT.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_DecisionTree.predict(x_test)
    # end_test = time.time()

    # print("Accuracy Decision Tree:", metrics.accuracy_score(y_test, prediction))
    print('R2 Score', metrics.r2_score(y_test, prediction))
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

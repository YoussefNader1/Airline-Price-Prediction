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

# Y.to_frame(name='my_column_name')
Y = pd.DataFrame(Y)

# Feature processing
X = X_preprocessData(X, data)
print(X)

# Feature selection
X = pd.DataFrame(X,
                 columns=['ch_code', 'num_code', 'time_taken', 'stop', 'type', 'month'])


print(X)

# Feature encoding
cols = ('ch_code', 'type')  # column to be encoded
X = Feature_Encoder(X, cols)
Y = Feature_Encoder_TicketCategory(Y)
type = X['type']

print(X)
X = pd.DataFrame(X,
                 columns=['ch_code', 'num_code', 'time_taken', 'stop', 'month'])
# Feature Scaling
X = featureScaling(X, 0, 1)

X = pd.concat([X, type], axis=1, join='inner')


print(X)

g = pd.Series(Y, name='TicketCategory')
Y = pd.DataFrame(g)




X.interpolate()
Y.interpolate()
X = np.array(X)
Y = np.array(Y)

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

    # loading linear model
    pickled_model_linear = pickle.load(open('lin.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------

    # start_test = time.time()
    prediction = pickled_model_linear.predict(X)
    # end_test = time.time()

    print("Accuracy linear:", metrics.accuracy_score(Y, prediction))
    print('R2 Score', metrics.r2_score(Y, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)


# elif choice == 2:
#
#     # loading polynomial model
#     pickled_model_polynomial = pickle.load(open('poly.pkl', 'rb'))
#     # ----------------------------------------------------------------------------------------------------
#     # start_test = time.time()
#     prediction = pickled_model_polynomial.predict(X)
#     # end_test = time.time()
#
#     print("Accuracy polynomial:", metrics.accuracy_score(Y, prediction))
#     print('R2 Score', metrics.r2_score(Y, prediction))
#     print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
#     # print("Actual time for training", endTrain - startTrain)
#     # print("Actual time for Testing", end_test - start_test)
#
# # RBF Model
# elif choice == 3:
#
#     # loading rbf model
#     pickled_model_rbf = pickle.load(open('rbf.pkl', 'rb'))
#     # ----------------------------------------------------------------------------------------------------
#     # start_test = time.time()
#     prediction = pickled_model_rbf.predict(X)
#     # end_test = time.time()
#
#     print("Accuracy rbf:", metrics.accuracy_score(Y, prediction))
#     print('R2 Score', metrics.r2_score(Y, prediction))
#     print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
#     # print("Actual time for training", endTrain - startTrain)
#     # print("Actual time for Testing", end_test - start_test)
#
#
# # Kernel Model
# elif choice == 4:
#
#     # loading Kernel model
#     pickled_model_kernel = pickle.load(open('kernel.pkl', 'rb'))
#     # ----------------------------------------------------------------------------------------------------
#     # start_test = time.time()
#     prediction = pickled_model_kernel.predict(X)
#     # end_test = time.time()
#
#     print("Accuracy kernel:", metrics.accuracy_score(Y, prediction))
#     print('R2 Score', metrics.r2_score(Y, prediction))
#     print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
#     # print("Actual time for training", endTrain - startTrain)
#     # print("Actual time for Testing", end_test - start_test)
#
# # Logistic Regression Model
# elif choice == 5:
#
#     # loading Logistic Regression model
#     pickled_model_LogisticRegression = pickle.load(open('LogisticRegression.pkl', 'rb'))
#     # ----------------------------------------------------------------------------------------------------
#     # start_test = time.time()
#     prediction = pickled_model_LogisticRegression.predict(X)
#     # end_test = time.time()
#
#     print("Accuracy Logistic Regression:", metrics.accuracy_score(Y, prediction))
#     print('R2 Score', metrics.r2_score(Y, prediction))
#     print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
#     # print("Actual time for training", endTrain - startTrain)
#     # print("Actual time for Testing", end_test - start_test)
#
# Decision Tree Model
elif choice == 6:

    # loading Decision Tree model
    pickled_model_DecisionTree = pickle.load(open('DT.pkl', 'rb'))
    # ----------------------------------------------------------------------------------------------------
    # start_test = time.time()
    prediction = pickled_model_DecisionTree.predict(X)
    # end_test = time.time()

    # print("Accuracy Decision Tree:", metrics.accuracy_score(Y, prediction))
    print('R2 Score', metrics.r2_score(Y, prediction))
    print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
    # print("Actual time for training", endTrain - startTrain)
    # print("Actual time for Testing", end_test - start_test)

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from Correlation import *
import time

# Load Airline data
data = pd.read_csv('airline-price-classification.csv')
# Features
X = data.iloc[:, 0:10]
# Label
Y = data['TicketCategory']

# Feature processing
X = X_preprocessData(X, data)
# Label processing
# Y = Y_preprocessData(Y)

# Feature encoding
cols = ('airline', 'ch_code', 'type', 'source', 'destination')  # column to be encoded
X = Feature_Encoder(X, cols)

# Feature Scaling
X = featureScaling(X, 0, 1)


Y = Feature_Encoder_TicketCategory(Y)

print(Y)

# Feature selection
top_features = correlation(X, Y)
X = X[top_features]



# # Data distribution
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,
#                                                     random_state=1)  # Features splitting
#
# # change dataframes into arrays
# X = np.array(X)
# Y = np.array(Y)
#
# # Drop nulls for train data
# x_train.dropna(how='any', inplace=True)
# y_train.dropna(how='any', inplace=True)
# # Replace null for test data
# x_test.fillna(0)
# y_test.fillna(0)



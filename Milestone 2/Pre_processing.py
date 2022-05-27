import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def Feature_Encoder_Type(X):
    l = []
    l = X['type']
    x = []
    for rows in range(len(l)):
        if l[rows] == 'business':
            x.append(0)
        elif l[rows] == 'economy':
            x.append(1)
        else:
            x.append(2)
    return x


def Feature_Encoder_ch_code(X):
    l = []
    l = X['ch_code']
    x = []
    for rows in range(len(l)):
        if l[rows] == '2T':
            x.append(0)
        elif l[rows] == '6E':
            x.append(1)
        elif l[rows] == 'AI':
            x.append(2)
        elif l[rows] == 'G8':
            x.append(3)
        elif l[rows] == 'I5':
            x.append(4)
        elif l[rows] == 'S5':
            x.append(5)
        elif l[rows] == 'SG':
            x.append(6)
        elif l[rows] == 'UK':
            x.append(7)
        # if uniqe value not found
        else:
            x.append(8)
    return x


def Feature_Encoder_TicketCategory(Y):
    l = []
    l = Y['TicketCategory']
    x = []
    for rows in range(len(l)):
        if l[rows] == 'very expensive':
            x.append(3)
        elif l[rows] == 'expensive':
            x.append(2)
        elif l[rows] == 'moderate':
            x.append(1)
        elif l[rows] == 'cheap':
            x.append(0)
    return x


def featureScalingTestScript(X, a, b):
    # min[0.0, 0.0, 101.0, 0.1, 0.5, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0, 2.0]
    # max[7.0, 7.0, 9991.0, 23.55, 49.5, 2.0, 23.59, 1.0, 5.0, 5.0, 31.0, 3.0]
    minn = [0, 101, 0.5, 0, 0, 2]
    maxx = [7, 9991, 49.5, 2, 1, 3]
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - minn[i]) / (maxx[i] - minn[i])) * (b - a) + a
    # As x is a np array, so we need to cast it into dataframe to avoid losing columns names
    DataFrameReturned = pd.DataFrame(Normalized_X,
                                     columns=['ch_code', 'num_code', 'time_taken', 'stop', 'type', 'month'])
    return DataFrameReturned


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    # As x is a np array, so we need to cast it into dataframe to avoid losing columns names
    DataFrameReturned = pd.DataFrame(Normalized_X,
                                     columns=['airline', 'ch_code', 'num_code', 'dep_time', 'time_taken', 'stop',
                                              'arr_time', 'type', 'source', 'destination', 'day', 'month'])
    return DataFrameReturned


def X_preprocessData(X, data):
    # PreProcessing
    # Stop Column
    X['stop'] = (X["stop"].str.split("-", n=1, expand=True))[0]
    X['stop'] = X['stop'].replace('non', '0')
    X['stop'] = X['stop'].replace('2+', '2')
    X['stop'] = pd.to_numeric(X['stop'])
    # dep_time Column
    X['dep_time'] = X['dep_time'].str.replace(':', '.')
    X['dep_time'] = pd.to_numeric(X['dep_time'])
    # arr_time Column
    X['arr_time'] = X['arr_time'].str.replace(':', '.')
    X['arr_time'] = pd.to_numeric(X['arr_time'])
    # route Column
    new = X["route"].str.split(", ")
    source = []
    destination = []
    for x in new:
        s = str(x[0]).split(": ")
        source.append(s[1])
        d = str(x[1]).split(": ")
        d = str(d[1]).split("}")
        destination.append(d[0])
    X = X.drop(['route'], axis=1)
    X['source'] = source
    X['destination'] = destination

    # timeTaken Column
    X['time_taken'] = X['time_taken'].str.replace('h m', '')
    X['time_taken'] = X['time_taken'].str.replace('h ', '.')
    X['time_taken'] = X['time_taken'].str.replace('m', '')
    X['time_taken'] = pd.to_numeric(X['time_taken'])

    # Date Column
    newDate = X['date']
    columnDate = []
    day = []
    month = []
    for i in range(len(data)):
        if newDate[i][-5] == "-":
            columnDate.append(datetime.strptime(newDate[i], '%d-%m-%Y'))
        elif newDate[i][-5] == "/":
            columnDate.append(datetime.strptime(newDate[i], '%d/%m/%Y'))
        day.append(columnDate[i].day)
        month.append(columnDate[i].month)

    X = X.drop(['date'], axis=1)
    X['day'] = day
    X['month'] = month

    X['day'] = pd.to_numeric(X['day'])
    X['month'] = pd.to_numeric(X['month'])

    return X

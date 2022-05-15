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

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    DataFrameReturned=pd.DataFrame(Normalized_X,columns=['airline','ch_code','num_code','dep_time','time_taken','stop','arr_time','type','source','destination','day','month'])
    return DataFrameReturned

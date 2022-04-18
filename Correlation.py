import pandas as pd

def correlation(X, Y):
    #Feature Selection
    #Get the correlation between the features

    data_corr = pd.DataFrame(X)
    data_corr = X
    result = pd.concat([data_corr, Y], axis=1, join='inner')
    result_corr = result.corr()
    #return only features with correlation more than 0.5
    top_feature = result_corr.index[abs(result_corr['price']) > 0.5]
    return top_feature

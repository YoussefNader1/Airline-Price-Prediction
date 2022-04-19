import pandas as pd

def correlation(X, Y):

    result = pd.concat([X, Y], axis=1, join='inner')
    result_corr = result.corr()
    # return only features with correlation more than 0.5
    top_feature = result_corr.index[abs(result_corr['price']) > 0.2]
    top_feature = top_feature.delete(-1)
    return top_feature

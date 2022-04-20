import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def correlation(X, Y):
    result = pd.concat([X, Y], axis=1, join='inner')
    result_corr = result.corr()
    # return only features with correlation more than 0.1
    top_feature = result_corr.index[abs(result_corr['price']) > 0.1]
    plt.subplots(figsize=(12, 8))
    top_corr = result[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    #drop top_feature[0] "airline" as it is highly correlated with "ch_code"
    top_feature = top_feature.delete(0)
    top_feature = top_feature.delete(-1)
    return top_feature

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def correlation(X, Y):

    result = pd.concat([X, Y], axis=1, join='inner')
    result_corr = result.corr()
    # return only features with correlation more than 0.5
    top_feature = result_corr.index[abs(result_corr['price']) > 0.2]
    plt.subplots(figsize=(12, 8))
    top_corr = result[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    return top_feature

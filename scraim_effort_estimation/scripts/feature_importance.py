from scraim_effort_estimation.modules import utils
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
 

def feature_importance(df, target='Effort', save_path=''):
    """
    Feature importance based on the random forest method

    Parameters:
    df: pandas datafame
    target (str): target column default = 'Effort' can also be 'Estimated time'
    save_path (str): save file path must be a .png file, if no path is specified does not save by default (save_path = '')

    Returns:
    plot feature importance graph, can save reulst to graph if a path is specified
    """     
    #feature_importance 
    if target == 'Effort':
        not_target = 'Estimated time'
    else:
        not_target = 'Effort'
    X = df.drop([target, not_target, 'Subject_clean','Project'], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
    feature_names = []
    for i in range(len(X.columns)):
        feature_names.append(X.columns[i])
    forest = RandomForestRegressor(n_estimators = 5000, 
                                        max_features =  'log2', 
                                        max_depth = 20)
    forest.fit(X_train, y_train)

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: "
        f"{elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(x=forest_importances.index, y=forest_importances.values)
    if save_path.endswith('.png'):
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__' :
    df = utils.read_from_storage('../storage/df_clean_wo_text.csv')
    feature_importance(df, save_path='../results/features_importance.png')
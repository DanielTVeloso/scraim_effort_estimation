from scraim_effort_estimation.modules import utils
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from tabulate import tabulate

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def parameter_tunning(df, target='Effort', save_path=''):
    """
    Parameter tunning based on gridsearchCV splits is set up to be 25% paramater tunning and using a 5 fold cv

    Parameters:
    df: pandas datafame
    target (str): target column default = 'Effort' can also be 'Estimated time'
    save_path (str): save file path must be a .txt file, if no path is specified does not save by default (save_path = '')

    Returns:
    print best parameters as a dataframe, can save reulst to file if save_path is especified
    """  
    #Hyperparameter tunning using GridSearchCV
    model_params = {
        'lasso' : {
            'model': Lasso(),
            'params': {
                'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            }
        },
        'svr' : {
            'model': SVR(),
            'params': {
                'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'kernel':['rbf', 'poly', 'sigmoid'],
                'gamma':[0.001, 0.01, 0.1, 1]
            }
        },
        'knn' : {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors':[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]       
            }
        },
        'extraTrees' : {
            'model': ExtraTreesRegressor(),
            'params': {
                'n_estimators':[100, 250, 500, 750, 1000],
                'max_features':['sqrt', 'log2', 'auto'],       
                'max_depth': [20, 50, 100, None]
            }
        },
        'random_forest' : {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators':[100, 250, 500, 750, 1000],
                'max_features':['sqrt', 'log2', 'auto'],       
                'max_depth': [20, 50, 100, None]       
            }
        },
        'gbm' : {
            'model': GradientBoostingRegressor(),
            'params': {
                'n_estimators':[250, 500, 750, 1000],
                'learning_rate':[0.0001, 0.001, 0.01, 0.2],
                'max_features':['sqrt', 'log2', 'auto'],
                'max_depth': [20, 50, 100, None] 
            }
        },
        'xgboost': {
            'model':XGBRegressor(),
            'params': {
                'learning_rate':[0.0001, 0.001, 0.01, 0.2],
                'max_depth': [20, 50, 100, None],
                'min_child_weight': [1, 4, 7],
                'subsample': [0.5, 0.75, 1],
                'gamma':[0, 0.2, 0.4],
                'colsample_bytree':[0.5, 0.75, 1]
            }
        },
        'mlp': {
            'model': MLPRegressor(), 
            'params':{
                'activation':['relu', 'tanh', 'logistic'],
                'hidden_layer_sizes':[[50, 100, 150],[150, 100, 50],[50, 150, 100],
                                    [150, 50, 100], [100, 50, 150], [100, 150, 50]],
                'alpha':[0.0001, 0.001, 0.01, 0.1, 1]
            }
        }
    }
    if target == 'Effort':
        not_target = 'Estimated time'
    else: #target = 'Estimated time'
        df.drop(['N of Part'], axis = 'columns', inplace=True) #if target is estimated time cannot use Number of participants
        not_target = 'Effort'
    #25% parameter tunning and 75% train with cv estimated time with text
    X = df.drop([target, not_target, 'Subject_clean','Project'], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=101)
    #X_train is for parameter training
    #X_test is used to train the model and test using cv
    #scaled data
    ss = StandardScaler()
    df_scaled = df.drop(['Subject_clean','Project'], axis= 'columns')
    df_scaled = pd.DataFrame(ss.fit_transform(df_scaled), columns = df_scaled.columns)
    X_s = df_scaled.drop([target, not_target], axis=1)
    y_s = df_scaled[target]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_s, y_s, test_size=0.75, random_state=101)
    scores = []
    for model_name, mp in model_params.items():
        print(model_name)
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        if (model_name == 'lasso') or (model_name == 'svr') or (model_name == 'knn'):
            clf.fit(Xs_train, ys_train)
        else:
            clf.fit(X_train, y_train) 
        scores.append({
            'model': model_name,
            'best_score': clf.best_score_,
            'best_params':clf.best_params_
        })
    df_grid_results = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    if save_path.endswith('.txt'):
        with open(save_path, 'w', encoding="utf-8") as f:
            f.write(tabulate(df_grid_results, headers='keys', tablefmt='fancy_grid', showindex="always", floatfmt=".4f"))
    print(df_grid_results)
if __name__ == '__main__' :
    df = utils.read_from_storage('../storage/df_tf_idf.csv')
    parameter_tunning(df, save_path='../results/parameter_tunning_effort.txt')
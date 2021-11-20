from nltk import util
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from statistics import mean
from tabulate import tabulate

from scraim_effort_estimation.modules import utils

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def leave_one_project_out(df, target = 'Effort', save_path=''):
    """
    Runs a specific cross validation which use all available projects as train and leave out for testing

    Parameters:
    df: pandas datafame
    target (str): target column default = 'Effort' can also be 'Estimated time'
    save_path (str): save file path must be a .txt file, if no path is specified does not save by default (save_path = '')

    Returns:
    print average r2 score of the mean prediction of all models, can save reulst to file if a path is specified
    """ 
    #define the models to be used on the evaluation
    estimators = [('ExtraTrees', ExtraTreesRegressor(n_estimators = 100, 
                                     max_features = 'log2', 
                                     max_depth = 100)),
               ('Random_Forest', RandomForestRegressor(n_estimators = 250, 
                                       max_features =  'sqrt', 
                                       max_depth = 100)), 
               ('GBM', GradientBoostingRegressor(n_estimators = 750,
                                           learning_rate = 0.01,
                                           max_features = 'sqrt',
                                           max_depth = 20 )),
               ('XGBoost', XGBRegressor(learning_rate = 0.2,
                             max_depth = 20,
                             min_child_weight = 1,
                             subsample = 0.5,
                             gamma = 0.2,
                             colsample_bytree = 0.75))]
    #leave one project out effort
    r2_scores = []
    for i in range(len(estimators)+1):
        r2_scores.append([])
    name_of_projects = df['Project'].unique()
    headers = ['id', 'Project_Name', 'r2_Extra', 'r2_Random', 'r2_GBM', 'r2_XGB', 'r2_Average']
    table = [headers]
    if target == 'Effort':
        not_target = 'Estimated time'
    else:
        not_target = 'Effort'
        df.drop(['N of Part'], axis = 'columns', inplace=True) #if target is estimated time cannot use Number of participants

    for i, element in enumerate(name_of_projects):
        X_train = df.loc[df['Project'] != name_of_projects[i]].drop([target, not_target, 'Subject_clean', 'Project'], axis=1)
        X_test = df.loc[df['Project'] == name_of_projects[i]].drop([target, not_target, 'Subject_clean', 'Project'], axis=1)
        y_train = df.loc[df['Project'] != name_of_projects[i]][target]
        y_test = df.loc[df['Project'] == name_of_projects[i]][target]
        y_pred = []
        for j in range(len(estimators)):
            estimators[j][1].fit(X_train, y_train)
            y_pred.append(estimators[j][1].predict(X_test))
            r2_scores[j].append(r2_score(y_test, y_pred[j]))
        y_average_pred = []
        for j in range(len(y_pred[0])):
            y_average_pred.append(((y_pred[0][j] + y_pred[1][j] + y_pred[2][j] + y_pred[3][j])/len(estimators))) 
        r2_scores[len(estimators)].append(r2_score(y_test, y_average_pred))
        my_line = []
        my_line.append(name_of_projects[i])
        for j in range(5):
            my_line.append(r2_scores[j][i])
        table.append(my_line)
    for i in range(len(estimators)):
        print('mean_r2_', estimators[i][0], '= ', mean(r2_scores[i]))
    print('mean_r2_Average', mean(r2_scores[i+1]))
    if save_path.endswith('.txt'):
        with open(save_path, 'w', encoding="utf-8") as f:
            f.write(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex="always", floatfmt=".4f"))

if __name__ == '__main__' :
    df = utils.read_from_storage('../storage/df_tf_idf.csv')
    leave_one_project_out(df, save_path='../results/leave_one_project_out.txt')
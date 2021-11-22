import pandas as pd
import numpy as np
from pandas.core.arrays import categorical
from scraim_effort_estimation.modules import utils
import nltk
import string
from nltk.corpus import stopwords
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate
from tabulate import tabulate
import joblib

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def cross_val_scores(df, target = 'Effort', save_path='', load_path='', verbose=False):
    """
    Get cross validation scores and save the results in a txt file if specified

    Parameters:
    df: pandas datafame
    target (str): target column default = 'Effort' can also be 'Estimated time'
    save_path (str): save file path must be a txt file, if no path is specified does not save by default (save_path = '')
    verbose (bool): If set True print information regarding to the cross_val process, default is False
    load_path(str): if load path is specified, load storaged models from that path

    Returns: can save models trained if save_path is especified, 
    can also print information regarding the training process if verbose is set to true
    """ 
    models_tunned = utils.model_tunned(target)
    #after 25% used for parameter tunning now 75% is used to train with cv estimated time with text
    if load_path == '':
        load_path = '../storage/models'
    if target == 'Effort':
        not_target = 'Estimated time'
        ss = joblib.load(load_path+'/effort_standard_scaler.joblib')
    else:
        not_target = 'Effort'
        df.drop(['N of Part'], axis = 'columns', inplace=True) #if target is estimated time cannot use Number of participants
        ss = joblib.load(load_path+'/effort_standard_scaler.joblib')
    X = df.drop([target, not_target, 'Subject_clean','Project'], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=101)
    #X_train é para parameter training
    #X_test é para treino do modelo e teste usando cv
    #ss = MinMaxScaler()
    X_s = pd.DataFrame(ss.transform(X), columns = X.columns)
    y_s = df[target]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_s, y_s, test_size=0.75, random_state=101)
    #selecionados os parâmetros tunados, fazer o treino nos 75% restantes usando cv de 5
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=101)
    scoring=('r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error')
    results = []
    for model_name, mp in models_tunned.items():
        if (model_name == 'lasso') or (model_name == 'svr') or (model_name == 'knn'): 
            #lasso, svr and knn need to use scaled data
            cv_results = cross_validate(mp['model'], Xs_test, ys_test, cv=kf, scoring=scoring, return_train_score=False)
            if verbose:
                print('')
                print(model_name)
                print(cv_results)
                print('r2 =', cv_results['test_r2'].mean())
                print('mae =', cv_results['test_neg_mean_absolute_error'].mean())
                print('rmse =', cv_results['test_neg_root_mean_squared_error'].mean())
                print('')
            results.append([model_name, cv_results['test_r2'].mean()])
        else:
            cv_results = cross_validate(mp['model'], X_test, y_test, cv=kf, scoring=scoring, return_train_score=False)
            if verbose:
                print('')
                print(model_name)
                print(cv_results)
                print('r2 =', cv_results['test_r2'].mean())
                print('mae =', cv_results['test_neg_mean_absolute_error'].mean())
                print('rmse =', cv_results['test_neg_root_mean_squared_error'].mean())
                print('')
            results.append([model_name, cv_results['test_r2'].mean()])
    headers = ['Method', 'r2_scores', 'rmse%']
    table = [headers, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
    if verbose:
        print('')
        print(target, 'results')
    if save_path.endswith('.txt'):
        with open(save_path, 'w', encoding="utf-8") as f:
            f.write(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex="always", floatfmt=".4f"))

def train_model(df, target = 'Effort', model='random_forest', save_path='', verobse=False):
    """
    Train model especified using all training data

    Parameters:
    df: pandas datafame
    target (str): target column default = 'Effort' can also be 'Estimated time'
    model (str): model defined by the user to be trained, default = 'random_forest' 
    save_path (str): save file path must be a folder, if no path is specified does not save by default (save_path = '')
    verbose (bool): If set True print information regarding to the cross_val process, default is False
    
    Returns: Can save trained model on a file if save_path is specified
    """ 
    models_tunned = utils.model_tunned(target)
    if target == 'Effort':
        target_s = 'effort_'
        not_target = 'Estimated time'
        X_train = df.drop([target, not_target, 'Subject_clean','Project'], axis=1)
        ss = joblib.load(save_path+'/effort_standard_scaler.joblib')
    else:
        not_target = 'Effort'
        target_s='est_time_'
        X_train = df.drop([target, not_target, 'Subject_clean','Project', 'N of Part'], axis=1)
        ss = joblib.load(save_path+'/est_time_standard_scaler.joblib')
    y_train = df[target]
    #scaled data
    ys_train = df[target] #does not scale the target
    Xs_train = pd.DataFrame(ss.transform(X_train), columns = X_train.columns)

    if model == 'all':
        for model_name, mp in models_tunned.items():
            if ((model_name == 'lasso') or (model_name == 'svr') or (model_name == 'knn')): 
                mp['model'].fit(Xs_train, ys_train)
                if save_path != '':
                    joblib.dump(mp['model'], save_path+'/'+target_s+model_name+'_trained.joblib') #save trained_model to disk
            else:
                mp['model'].fit(X_train, y_train)
                if save_path != '':
                    joblib.dump(mp['model'], save_path+'/'+target_s+model_name+'_trained.joblib') #save trained_model to disk
    else:
        for model_name, mp in models_tunned.items():
            if (model==model_name) & ((model_name == 'lasso') or (model_name == 'svr') or (model_name == 'knn')): 
                mp['model'].fit(Xs_train, ys_train)
                if save_path != '':
                    joblib.dump(mp['model'], save_path+'/'+target_s+model_name+'_trained.joblib') #save trained_model to disk
            elif (model==model_name):
                mp['model'].fit(X_train, y_train)
                if save_path != '':
                    joblib.dump(mp['model'], save_path+'/'+target_s+model_name+'_trained.joblib') #save trained_model to disk

def predict_from_model(request_data, load_path=''):
    """Predict results using the previously trained models (random_forest as default) and
    the data received in the request
    
    Parameters:
        request_data(json): Data from POST request
        load_path(str): if load path is specified, load storaged models from that path
    
    Returns:
        results(json): Results of the predictions
        status_code(int): HTTP status code to return
    """
    # check request data for errors before proceeding
    has_error, error_messages = utils.check_request_data(request_data)
    if has_error:
        return {
            'error': has_error,
            'messages': error_messages
        }, 400
    #load effort encoder / load standard sccaler and model
    if load_path == '':
        load_path = '../storage/models'
    ohe = joblib.load(load_path+'/one_hot_encoder.joblib')
    vectoriser = joblib.load(load_path+'/vectoriser.joblib')
    #using the storage one hot encoder
    private = []
    for i in range(len(request_data['Private'])):
        if (request_data['Private'][i] == 'No'):
            private.append('Not Private')
        else: 
            private.append('Private')
    categorical_features = {
        'Private': private,
        'Tracker': request_data['Tracker'],
        'Priority': request_data['Priority']
    }
    df_categorical_features = pd.DataFrame(categorical_features)
    df_categorical_features = ohe.transform(df_categorical_features)
    #after doing the encoder using the storaged_tfidf
    subject = {
        'Subject' : request_data['Subject']
    }
    df_subject = pd.DataFrame(subject)
    def remove_punc_stopwords(mess):
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        nopunc = nopunc.lower()
        return [word for word in nopunc.split() if word.lower() not in (stopwords.words('portuguese') or stopwords.words('english'))]
    df_subject['Subject_clean'] = df_subject['Subject'].apply(remove_punc_stopwords)
    #lemmatization => changes into change
    lemmatizer = nltk.stem.WordNetLemmatizer()
    def lemmatize_text(token_list):
        return [lemmatizer.lemmatize(w) for w in token_list]
    df_subject['Subject_clean'] = df_subject['Subject_clean'].apply(lemmatize_text)
    def list_to_string(lista):
        new_string = " ".join(lista)
        return new_string
    df_subject['Subject_clean'] = df_subject['Subject_clean'].apply(list_to_string)
    df_subject.drop('Subject', inplace = True, axis='columns')
    x = vectoriser.transform(df_subject['Subject_clean'])
    df_subject = pd.DataFrame(x.toarray(), columns=vectoriser.get_feature_names())
    #finally combining all the data into a final df
    n_of_part = {
        'N of Part' : request_data['N of Part']
    }
    df_n_of_part = pd.DataFrame(n_of_part)
    pdList = [df_n_of_part, df_categorical_features, df_subject]  # List of dataframes
    final_df = pd.concat(pdList, axis = 'columns')
    #with the final_df now need to load the saved model and predict
    model_name = request_data['Model']
    if request_data['Target'] == 'Effort':
        loaded_model = joblib.load(load_path+'/effort_'+model_name+'_trained.joblib')
        ss = joblib.load(load_path+'/effort_standard_scaler.joblib')
    else:
        final_df = final_df.drop(['N of Part'], axis= 'columns')
        loaded_model = joblib.load(load_path+'/est_time_'+model_name+'_trained.joblib')
        ss = joblib.load(load_path+'/est_time_standard_scaler.joblib')
    if ((model_name == 'lasso') or (model_name == 'svr') or (model_name == 'knn')): 
        #if lasso, svr or knn need to use scaled data
        final_df = pd.DataFrame(ss.transform(final_df), columns = final_df.columns)

    predictions = loaded_model.predict(final_df)
    predictions = predictions.tolist()

    results = {
        request_data['Target'] : predictions
    }
    return results, 200

if __name__ == '__main__' :
    #df = utils.read_from_storage('../storage/df_tf_idf.csv')
    #cross_val_scores(df, save_path='../results/cross_val_scores_effort.txt')
    #train_model(df, target='Estimated time', model = 'all', save_path='../storage/models')
    #train_model(df, target='Effort', model = 'all', save_path='../storage/models')
    request_data = {
               "N of Part" : [2, 2],
               "Project": ["My Test Project", "My Test Project"],
               "Subject": ["initial training", "initial evaluation"],
               "Private": ["Yes", "Yes"],
               "Tracker" : ["Task", "Task"],
               "Priority" : ["Trivial", "Trivial"],
               "Target" : "Effort",
               "Model" : "random_forest"
     }
    results = predict_from_model(request_data)
    print(results)
import glob
from xml.etree.ElementTree import ElementTree
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from category_encoders.one_hot import OneHotEncoder
import joblib

def combine_all_csv_files(path):
    """
    Read all the csv files from the folder data and combines it on a dataframe
    
    Parameters:
    path (str): path of the folder containing the csv files
    
    Returns:
    df: pandas dataframe
    """ 
    #read all csv files
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, encoding = "ISO-8859-1")
        li.append(df)
    #concatenate all files into one dataframe
    df = pd.concat(li, axis=0, ignore_index=True)
    return df

def clean_missing_data(df, threshold=70):
    """
    Removes empty rows and columns from a df,
    Additonaly removes columns with a (%) of missing rows and columns 
    
    Parameters:
    df: pandas datafame
    threshold (int): (%) of missing value in a row/column default = 70%
    
    Returns:
    df: pandas dataframe
    """
    #drop completely empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(how='all', inplace=True, axis='columns')     
    #drop columns with more than threshold% empty values
    df.dropna(thresh=df.shape[0]*(threshold/100),how='all',axis=1, inplace=True)
    return df

def data_pre_processing(df, save_path=''):
    """
    Does all the df pre_processing:
    Filters the df with only closed tasks, that have target and also have participants
    Turn the names of participants into number of participants
    Based on the estimated time and number of participants calculate the effort
    Drop any unusable and unneccessary columns and resets index
    Drop outliers projects
    Convert categorical features into numerical ones 

    Parameters:
    df: pandas datafame
    save_path (str): save file path must be a folder, if no path is specified does not save by default (save_path = '')
    
    Returns:
    df: pandas dataframe, can also save one hot encoder model if save_path folder is specified
    """ 
    if 'Status' in df.columns: #depend on the df that was passed
        #can only work with resolved/closed tasks
        df = df.loc[df['Status'].isin(['Resolved','Closed'])]
        #can only work with labeled tasks 
        df = df.dropna(subset=['Estimated time'])
        #can only work with one particpants or more
        df = df.dropna(subset=['Participants'])
        #Convert name of participants into number of participants
        participants = {}
        unique_elements = df['Participants'].unique()
        df['N of Part'] = df['Participants']
        for element in unique_elements:
            # add number of participants to the dictionary
            try:
                count = element.count(",") + 1
                participants[element] = count
            except:
                participants[element] = 0
        df['N of Part'].replace(participants, inplace = True)
        #Based on the number of participants and estimated time calculates the effort
        #Effort = number of hours * number of participants
        df['Effort'] = df['Estimated time'] * df['N of Part']
        #drop all clomuns not capable of being features
        df.drop(['Updated','Created', '% Done', 'Author', 'Target version', 'Centro de custo', 
            'Status', 'Total estimated time', 'Total spent time', 'Iteration', 'Assignee', 'Last updated by',
            'Spent time', 'Start date', 'Due date', '#', 'Participants', 'Closed', 'id', 'Subject'],axis='columns', inplace=True)
        #need to reset index
    #Need to remove estimated = 0
    df = df.loc[df['Estimated time'] != 0]
    df.reset_index(drop=True, inplace=True)
    #trying to remove the name of the project from descpritions
    descripitons_wo_proj_name = {}
    descriptions = df['Subject_clean']
    for i, element in enumerate(descriptions):
        if 'nomad tech' in element:
    #        my_string_testin = element.replace('nomad tech', '')
            descripitons_wo_proj_name[element] = element.replace('nomad tech', '')
        elif 'deloitte' in element:
            descripitons_wo_proj_name[element] = element.replace('deloitte', '')
        elif 'ebankit' in element:
            descripitons_wo_proj_name[element] = element.replace('ebankit', '')
        elif 'g9' in element:
            descripitons_wo_proj_name[element] = element.replace('g9', '')
        elif 'procensus' in element:
            descripitons_wo_proj_name[element] = element.replace('procensus', '')
        elif 'dodoc' in element:
            descripitons_wo_proj_name[element] = element.replace('dodoc', '')    
        elif 'xpand it' in element:
            descripitons_wo_proj_name[element] = element.replace('xpand it', '')
        else:
            descripitons_wo_proj_name[element] = [element]
    df['Subject_clean'].replace(descripitons_wo_proj_name, inplace=True)    
    #drop outdated/outliers projects
    projects_to_remove = ['106.2 Effizency - RGPD', 
                        '40.5 Innowave - RGPD', 
                        '40.6 InnoWave - CMMI Development 3 Renovação', 
                        '85.1 Ebankit - ISO 27001', 
                        '87.1 Frotcom ITMark',
                        '91.1 Médicos no Mundo - RGPD',
                        '99.1 doDOC - ISO 27001'
                    ]
    df = df.loc[~df['Project'].isin(projects_to_remove)]
    # need to reset index after removing rows
    df.reset_index(drop=True, inplace=True)
    #need to transform categorical features into numerical ones ----> get dummies
    def private_dummy(element):
        #convert yes or no into private or not private
        if (element == 'No'):
            element = 'Not Private'
        else: #(element == 'Yes'):
            element = 'Private'
        return element
    def priority_dummy(element):
        #convert minor or major into trivial or critical
        if (element == 'Minor'):
            element = 'Trivial'
        elif (element == 'Major'): 
            element = 'Critical'
        return element
    #using and saving, if specified one hot econder model
    df['Private'] = df['Private'].apply(private_dummy)
    df['Priority'] = df['Priority'].apply(priority_dummy)
    df_enc = df.drop(['Project', 'Estimated time', 'Effort', 'N of Part', 'Subject_clean'], axis = 'columns') 
    cols_encoding = df_enc.select_dtypes(include='object').columns
    ohe = OneHotEncoder(cols=cols_encoding)
    encoded = ohe.fit_transform(df_enc) 
    if save_path != '':
            joblib.dump(ohe, save_path+'/one_hot_encoder.joblib') 
    df = pd.concat([df.drop(['Private', 'Tracker', 'Priority'], axis = 'columns'), encoded], axis = 1)
    df.head()

    return df

def text_processing(df, method='TF-IDF', save_path=''):
    """
    Removes punctuation, stopwords 
    Does lemmatization
    Conver text into numerical features based on the method choosen

    Parameters:
    df: pandas datafame

    method (str): method used to convert the text into (default = 'TF-IDF') can be = ['TF-IDF', 'None']
    save_path (str): save file path must be a folder, if no path is specified does not save by default (save_path = '')

    Returns:
    df: pandas dataframe, can also save vectorizer from tf-idf and also standard scaler if save_path folder is especified
    """

    def remove_punc_stopwords(mess):
        """
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]

        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        nopunc = nopunc.lower()
    
        # Now just remove any stopwords
        return [word for word in nopunc.split() if word.lower() not in (stopwords.words('portuguese') or stopwords.words('english'))]
    if 'Subject' in df.columns:
        df['Subject_clean'] = df['Subject'].apply(remove_punc_stopwords)
        #lemmatization => changes into change
        lemmatizer = nltk.stem.WordNetLemmatizer()
        def lemmatize_text(token_list):
            return [lemmatizer.lemmatize(w) for w in token_list]
        df['Subject_clean'] = df['Subject_clean'].apply(lemmatize_text)
        def list_to_string(lista):
            new_string = " ".join(lista)
            return new_string
        df['Subject_clean'] = df['Subject_clean'].apply(list_to_string)
        df.drop('Subject', inplace = True, axis='columns')
    if (method == 'TF-IDF'):
        #TF-IDF
        vectoriser = TfidfVectorizer()
        x = vectoriser.fit_transform(df['Subject_clean'])
        df1 = pd.DataFrame(x.toarray(), columns=vectoriser.get_feature_names())
        df = pd.concat([df, df1], axis='columns')
        ss_effort = StandardScaler() #effort use N of Part
        ss_est_time = StandardScaler() #est_time does not use N of Part
        df_effort_scaled = df.drop(['Subject_clean','Project', 'Estimated time', 'Effort'], axis= 'columns')
        df_est_time_scaled = df.drop(['Subject_clean','Project', 'N of Part', 'Effort', 'Estimated time'], axis= 'columns')
        df_effort_scaled = pd.DataFrame(ss_effort.fit_transform(df_effort_scaled), columns = df_effort_scaled.columns)
        df_est_time_scaled = pd.DataFrame(ss_est_time.fit_transform(df_est_time_scaled), columns = df_est_time_scaled.columns)
        #If specified, save tf-idf vectoriser and also Standard Scaler
        if save_path != '':
            joblib.dump(vectoriser, save_path+'/vectoriser.joblib') 
            joblib.dump(ss_effort, save_path+'/effort_standard_scaler.joblib')
            joblib.dump(ss_est_time, save_path+'/est_time_standard_scaler.joblib')
            df.to_csv('../storage/df_tf_idf.csv', encoding='utf-8', index = False)
    return df

def read_from_storage(path):
    """
    Read df from storage 

    Parameters:
    path (str): Path of the file to read and load to the csv
    
    Returns:
    df: pandas dataframe
    """
    df = pd.read_csv(path, index_col=None, header=0, encoding = "ISO-8859-1")
    return df

def model_tunned(target='Effort'):
    """
    Simple function that return a dict of models with the best parameters defined by the paramater_tunning program

    Parameters:
    target (str): target used to define the best parameters
    
    Returns:
    models_tunned (dict): dictionary cointaning the the name of the model and models 
    """
   #the best parameters for each model were selected from the parameter_tunning script
    if target == 'Effort':
        models_tunned = {
            'lasso' : {
                'model': Lasso(alpha = 0.1)
            },
            'svr' : {
                'model': SVR(C = 100, kernel = 'rbf', gamma = 0.01)
            },
            'knn' : {
                'model': KNeighborsRegressor(n_neighbors = 5)
            },
            'extratrees' : {
                'model': ExtraTreesRegressor(n_estimators = 1000, 
                                            max_features = 'log2', 
                                            max_depth = 20)
            },
            'random_forest' : {
                'model': RandomForestRegressor(n_estimators = 250, 
                                            max_features =  'log2', 
                                            max_depth = 50)
            },
            'gbm' : {
                'model': GradientBoostingRegressor(n_estimators = 1000,
                                                learning_rate = 0.001,
                                                max_features = 'sqrt',
                                                max_depth = 100 )
            },
            'xgboost': {
                'model':XGBRegressor(learning_rate = 0.2,
                                    max_depth = 50,
                                    min_child_weight = 1,
                                    subsample = 0.5,
                                    gamma = 0.2,
                                    colsample_bytree = 1)
            },
            'mlp': {
                'model': MLPRegressor(activation = 'relu', 
                                    hidden_layer_sizes =[50, 150, 100],
                                    alpha = 0.1) 
            }
        }
    else: #target = 'Estimated time'
        models_tunned = {
            'lasso' : {
                'model': Lasso(alpha = 0.4)
            },
            'svr' : {
                'model': SVR(C = 1, kernel = 'rbf', gamma = 0.1)
            },
            'knn' : {
                'model': KNeighborsRegressor(n_neighbors = 5)
            },
            'extratrees' : {
                'model': ExtraTreesRegressor(n_estimators = 500, 
                                            max_features = 'log2', 
                                            max_depth = 20)
            },
            'random_forest' : {
                'model': RandomForestRegressor(n_estimators = 250, 
                                            max_features =  'log2', 
                                            max_depth = 20)
            },
            'gbm' : {
                'model': GradientBoostingRegressor(n_estimators = 750,
                                                learning_rate = 0.001,
                                                max_features = 'auto',
                                                max_depth = 20 )
            },
            'xgboost': {
                'model':XGBRegressor(learning_rate = 0.2,
                                    max_depth = 100,
                                    min_child_weight = 1,
                                    subsample = 1,
                                    gamma = 0.4,
                                    colsample_bytree = 1)
            },
            'mlp': {
                'model': MLPRegressor(activation = 'logistic', 
                                    hidden_layer_sizes =[50, 150, 100],
                                    alpha = 0.0001) 
            }
        }
    return models_tunned

def check_request_data(data):
    """Check request data for consistency and correctness
    Args:
        data: Request data
    Returns:
        has_error: Whether there is an error in the request data
        messages: Error messages to display
    """

    has_error = False
    messages = []
    mandatory_field_msg = '%s is a mandatory field.'
    request_properties = ['N of Part', 'Project', 'Subject', 'Private', 'Tracker', 'Priority', 'Target', 'Model']

    for x in request_properties:
        if x not in data or data[x] == '':
            has_error = True
            messages.append(mandatory_field_msg % x)

    return has_error, messages

if __name__ == '__main__' :
    df = read_from_storage(path = '../storage/all_translated_2.csv')
    df = data_pre_processing(df, save_path='../storage/models')
    df = text_processing(df, save_path='../storage/models')
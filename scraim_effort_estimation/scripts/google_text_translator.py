#!/usr/bin/python
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from googletrans import Translator  

from scraim_effort_estimation.modules import utils


def text_translation(df, lang_dest='en', verbose=False, save_path =''):
    """
    Translate text from one language to another unsing google trans

    Parameters:
    df: pandas datafame
    lang_dest (str): target language to translate (default = 'en')
    verbose (bool): text feedback
    save_path(str): save file path must a .csv file, if no path is specified does not save by default (save_path = '')

    Returns:
    df: pandas dataframe
    """ 

    #converting sentences to english because the lemmatization process only works with english words
    translator = Translator()
    translations = {}
    convertidos = 0
    nao_convertidos = 0
    unique_elements = df['Subject_clean'].unique()
    for element in unique_elements:
        # add translation to the dictionary
        #try is necessary because google trasnlator sometimes can't translate
        #because of problems like to much words to translate at a time
        try:
            translations[element] = translator.translate(element, dest=lang_dest).text
            convertidos = convertidos + 1
        except:
            translations[element] = element
            nao_convertidos = nao_convertidos + 1
    if verbose:           
        print('translations:')
        print(translations)
        print('convertidos = ', convertidos)
        print('nao_convertidos = ', nao_convertidos) 
    df['Subject_clean'].replace(translations, inplace = True) 
    if save_path.endswith('.csv'):
        df.to_csv(save_path, encoding='utf-8')
    return df 

def text_lang_test(df, lang_test='en'):
    """
    Test if the text is written one especified language or not

    Parameters:
    df: pandas datafame
    lang_test (str): target language to translate (default = 'en')

    Returns:
    """ 
    translator = Translator()
    convertidos = 0
    nao_convertidos = 0
    lang_pt = 0
    lang_en = 0
    unique_elements = df['Subject_clean'].unique()
    for element in unique_elements:
        try:
            if (translator.detect(element).lang == 'en'):
                lang_en = lang_en + 1
            else:
                lang_pt = lang_pt + 1
            convertidos = convertidos + 1
        except:
            nao_convertidos = nao_convertidos + 1  
    print('sucess = ', convertidos)
    print('failed = ', nao_convertidos)
    print('lang_pt = ', lang_pt)
    print('lang_en = ', lang_en)    

if __name__ == '__main__' :
    df = utils.combine_all_csv_files(path = '../data/2021_2020_with_process')
    df = utils.clean_missing_data(df)
    df = utils.data_pre_processing(df)
    df = utils.text_processing(df)
    text_translation(df, verbose=True, save_path='../results/df_translated.csv')
    text_lang_test(df)


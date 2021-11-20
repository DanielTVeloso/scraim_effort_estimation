#!/usr/bin/python

from scraim_effort_estimation.modules import model
from scraim_effort_estimation.modules import utils
from scraim_effort_estimation.modules import server

if __name__ == '__main__':
    df = utils.read_from_storage('storage/df_tf_idf.csv') 
    #train both models
    model.train_model(df, target='Effort', model = 'all', save_path='storage/models')
    model.train_model(df, target='Estimated time', model = 'all', save_path='storage/models')

    #run Flask server
    server.run_server()

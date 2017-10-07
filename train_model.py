import config
import model_stats

import os, sys, yaml, logging
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib
from keras.optimizers import SGD

from keras.models import Model, Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def get_train_test(X, Y, year):
    # Remove old years
     
    # Train Test Split for Features
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=42)
    return (X_train, Y_train, X_test, Y_test)

def add_features(df, year, zipcode):

    df['zipcode'] = get_zipcode(df['long'], df['lat'])
    
    return df
   
def convert_data_to_series(X, Y):

    methods = ['','ineligible','novote_eligible','method_early','method_absentee']
    prim_or_gen = [0, 1]
    p_or_gs = ['p', 'g']
    temps = []
    for year in range(1998, 2016, 1) : 
        presidential = (year in presidential_years)
        odd_spec = (year in odd_years)
        for p_or_g in prim_or_gen:
            col = "vote_%s%i" % (p_or_gs[p_or_g], year)
            if col in X: 
                temp = pd.DataFrame(X.index, columns=['personid'], index=X.index)
                temp['year'] = year # for reference
                temp['general'] = p_or_g
                temp['vote'] = X[col]
                temp['eligible'] = (X[col+"_ineligible"] == 0)*1.0
                temp['presidential'] = int(presidential)
                temp['special'] = int(not odd_spec)
                temp['age'] = X.age_golden + year - 2016
                temps.append(temp)
            
    data_series = pd.concat(temps, ignore_index=True)
    
    input_cols = ["year", "general", "vote", "eligible", "presidential", "special", "age"]
    output_cols = ["vote"]
    ds = pd.DataFrame() 
    
    data_series['single_input_vector'] = data_series[input_cols].apply(tuple, axis=1).apply(list)
    ds['single_input_vector'] = data_series.groupby('personid')['single_input_vector'].apply(lambda x: list(x))
       
    max_sequence_length = len(ds.single_input_vector.iloc[0])
    X_init = np.asarray(ds.single_input_vector)
    
    X_ds = np.hstack(X_init).reshape(len(ds),max_sequence_length,len(input_cols))
    Y_ds = np.hstack(np.asarray(Y)).reshape(len(ds),len(output_cols))

    return (X_ds, Y_ds)


    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), numpy.array(dataY)  
    


def train(X, Y, target_year, hptuning, job_folder):
    """Trains a model. Note that this is *only* the training step. This method expects data that has already 
    gone through preprocessing and feature engineering.

    Args:
      df: Dataframe of data that has gone through preprocessing and feature engineering
      model_type: Type of model to train.
      hptuning: Hyperparameter tuning paramgrid.
      job_folder: File to export saved model to.
    """
    target_year = 2017
    X_train, Y_train, X_test, Y_test = get_train_test(X, Y, target_year)        
    
    X_train_ds, Y_train_ds = convert_data_to_series(X_train, Y_train)
    X_test_ds, Y_test_ds = convert_data_to_series(X_test, Y_test)
    
    
    joblib.dump(X_train_ds, os.path.join(job_folder, 'model_xtrain.pkl')) 
    joblib.dump(Y_train_ds,os.path.join(job_folder, 'model_ytrain.pkl'))
    joblib.dump(X_test_ds, os.path.join(job_folder, 'model_xtest.pkl'))
    joblib.dump(Y_test_ds, os.path.join(job_folder, 'model_ytest.pkl'))
    
    
    input_length = X_train_ds.shape[1]
    input_dim = X_train_ds.shape[2]
    output_dim = len(Y_train_ds[0])

    momentum = 0
    
    # normalize the dataset ? 
    # scaler = MinMaxScaler(feature_range=(0, 1))    
    # X_train_ds = scaler.fit_transform(X_train_ds)
    # X_test_ds = scaler.fit_transform(X_test_ds)
    
    def create_model(learn_rate=0.01, units=8):
        # Build the model
        model = Sequential()
    
        model.add(LSTM(units=8, input_shape=(input_length, input_dim)))
        # The max output value is > 1 so relu is used as final activation.
        model.add(Dense(output_dim, activation='sigmoid'))
        optimizer = SGD(lr=learn_rate, momentum=momentum)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
#        history = model.fit(X_train_ds, Y_train_ds,
#                            batch_size=10, epochs=100,
#                            verbose = 1)
        
        return model

    model = KerasClassifier(build_fn=create_model, epochs=3, verbose=1, batch_size=10)
    
    hptuning = {'learn_rate': [.01, 0.1, 0.3], 'units':[8,16,32]}

    grid_clf = GridSearchCV(estimator=model, param_grid=hptuning,
                            scoring='accuracy', verbose=50, n_jobs=-1)
    grid_result = grid_clf.fit(X_train_ds, Y_train_ds)

    best_clf = grid_clf.best_estimator_

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

     # Save Model
    model_file_name = 'model_%s.pkl' % (model_type)
    joblib.dump(best_clf, os.path.join(job_folder, model_file_name))
    
    
    model_stats.show_model_stats(best_clf, X_train_ds, Y_train_ds.flatten(), X_test_ds, Y_test_ds.flatten())

    return best_clf

    # Evaluate model
    ##TODO model_stats.show_model_stats(best_clf, X_train, Y_train, X_test, Y_test)
def for_spyder_debugging():
    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    root_dir = "/Users/homaei/Documents/Projects/FBMLHackthon2017/fbmlhackthon"
    return Namespace(hptuning={'n_estimators': [10, 100]}, hptuning_config=root_dir+'/configs/hptuning/simple_randomforest.yaml', input='/Users/homaei/Documents/Projects/WashDems/WADemAnalysis/data/interim/100k_sample.pkl', input_type='FeatureEngineered', job_folder='/Users/homaei/Documents/Projects/WashDems/WADemAnalysis/jobs/job_20170811', model_input=None, model_type='LSTM', project_folder='/Users/homaei/Documents/~/Documents/Projects/FBMLHackthon2017/fbmlhackthon', sample_size='100k', target_election='p2016', verbosity='DEBUG')
   
def main():

    start_time = time.time()

    """Reads configuration and trains a model."""
    
    options = config.get_config()
   # options = for_spyder_debugging()
    if options.input_type != 'FeatureEngineered':
        raise RuntimeError('train_model.py expects input that has already been preprocessed and feature engineered. Consult the README.')

    logging.debug('Reading input %s' % options.input)
    X = pd.read_csv(options.input)
    Y = pd.read_pickle(options.output)

    logging.debug('Finished reading %s' % options.input)

    train(X, Y, options.model_type, options.hptuning, options.job_folder)
    
    print("--- Execution time:  %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()

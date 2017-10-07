import os, sys, yaml, logging
from time import strftime
import configargparse

def get_config():

    PROJECTFOLDER = os.getcwd()[0:os.getcwd().find('fbmlhackthon')+13]


    p = configargparse.ArgParser(default_config_files=[os.path.join(PROJECTFOLDER,'configs','default.txt')])
    p.add('--verbosity', help='Logging verbosity', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'], default='INFO')
    p.add('--job_folder', help='Folder for job output', default=os.path.join(PROJECTFOLDER, 'jobs', 'job_%s' % (strftime("%Y%m%d_%H%M"))))

    # Configuration for both training and prediction
    p.add('--input', help='Input to train or predict. Typically a folder containing five raw files dumped from Vertica.', default=os.path.join(PROJECTFOLDER,'data','raw'))
    p.add('--input_type', help='Allows for skipping preprocessing or feature engineering with interim data', choices=['Raw', 'Preprocessed', 'FeatureEngineered'])

    # Configuration for training only
    p.add('--model_type', help='Type of model to train.', choices=['RandomForest', 'XGBoost', 'LSTM'])
    p.add('--hptuning_config', help='HyperParameter Tuning configuration.', default=os.path.join(PROJECTFOLDER,'configs', 'hptuning', 'lstm.yaml'))

    # Configuration for prediction only
    p.add('--model_input', help='For Prediction, load a saved model from this location.')

    options = p.parse_args()
    options.project_folder = PROJECTFOLDER
    folder_map = {'Raw': 'raw',
                'Preprocessed': 'interim',
                'FeatureEngineered': 'interim'}

    options.input = os.path.join(PROJECTFOLDER,'data', 
                                 folder_map.get(options.input_type), 
                                 file_map.get(options.sample_size))    
    
    if not os.path.exists(options.job_folder):
        os.makedirs(options.job_folder)

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=options.verbosity, format=log_fmt)

    with open(options.hptuning_config) as info:
        options.hptuning =  yaml.load(info)

    return options

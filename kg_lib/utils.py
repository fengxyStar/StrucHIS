import os
import json
import yaml

from datetime import date, datetime, timedelta


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def read_json_config_file(config_filename):
    with open(config_filename, 'r') as f:
        config = json.loads(f.read()) 
    
    return config



    
# load config file
def load_config(config_filepath):
    config_dict = {}
    if os.path.exists(config_filepath):
        with open(config_filepath,'rb') as f:
            config_dict = yaml.safe_load(f.read())    
    else:
        raise ValueError('Input path does not exist!')
    return config_dict
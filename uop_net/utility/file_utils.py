import os
from os.path import join, isfile, isdir, splitext
from os import listdir
import yaml
from datetime import datetime
import logging
import shutil
import json
import pickle
import time

# logger
def get_logger(module_name):
    logger = logging.getLogger(module_name)
    # formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    # streamHandler = logging.StreamHandler()
    # streamHandler.setFormatter(formatter)
    # logger.addHandler(streamHandler)
    logger.setLevel(level=logging.INFO)
    return logger



# os file functions
def get_file_list(path):
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    return file_list

def get_dir_list(path):
    dir_list = [join(path, f) for f in listdir(path) if isdir(join(path, f))]

    return dir_list

def get_whole_list(path):
    whole_file_list = []
    if isdir(path):
        for file_path in [join(path, f) for f in listdir(path)]:
            whole_file_list += get_whole_list(file_path)
    elif isfile(path):
        whole_file_list += [path]
    else:
        pass

    return whole_file_list


def get_dir_name(path):
    return path.split("/")[-1]

def get_file_name(path):
    file_path, _ = splitext(path)
    return file_path.split("/")[-1]

def get_dir_path(path):
    return os.path.dirname(path)

def check_and_create_dir(dir_path):
    if not check_dir(dir_path):
        os.mkdir(dir_path)
        return True
    else:
        return False

def check_and_reset_dir(dir_path):
    if check_dir(dir_path):
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        return True
    else:
        os.mkdir(dir_path)
        return False

def check_dir(dir_path):
    return os.path.isdir(dir_path)

def check_file(file_path):
    return os.path.isfile(file_path)

def relative_path_to_abs_path(rel_path):
    os.path.abspath(rel_path)
    return os.path.abspath(rel_path)

def remove_file(file_path):
    os.remove(file_path)
def remove_dir(dir_path):
    shutil.rmtree(dir_path)

def copy_file(src, target):
    shutil.copy(src, target)

#============== experiment logging
def get_time_stamp():
    return datetime.timestamp(datetime.now())

def get_current_time() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def create_exp_dir(path, scripts_to_save=None):
    """reference:https://github.com/quark0/darts
    Usage: 
        create_exp_dir(save_path, scripts_to_save=glob.glob('*.py'))
    Args:
        path ([type]): [description]
        scripts_to_save ([type], optional): [description]. Defaults to None.
    """
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)




#============== Specific file format
class FileFormat: #TODO
    yaml_format = []
    json_format = []
    pickle_format = []
    
# yaml 
def save_dic_to_yaml(dic, yaml_path):
    with open(yaml_path, 'w') as y_file:
        _ = yaml.dump(dic, y_file, default_flow_style=False)

def load_yaml_to_dic(yaml_path):
    with open(yaml_path, 'r') as y_file:
        dic = yaml.load(y_file, Loader=yaml.FullLoader)
    return dic



# json
def load_json_to_dic(json_path):
    with open(json_path, 'r') as j_file:
        dic = json.load(j_file)
    return dic

def save_dic_to_json(dic, json_path):
    with open(json_path, 'w') as j_file:
        json.dump(dic, j_file, sort_keys=True, indent=4)



# pickle
def save_to_pickle(data, pickle_path):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except ValueError:
            import pickle5
            data = pickle5.load(f)

    return data
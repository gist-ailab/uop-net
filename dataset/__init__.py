import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import os
from utils.file_utils import *
from dataset.uopsim import UOPSIM


#TODO: path to dataset
DATA_DIR=os.path.join(os.environ['UOPROOT'], 'uop_data')
YCB_PATH = '{}/ycb'.format(DATA_DIR)
THREEDNET_PATH = '{}/3dnet'.format(DATA_DIR)
SHAPENET_PATH = '{}/shapenet'.format(DATA_DIR)

val_split = load_json_to_dic( os.path.join( os.path.dirname(__file__), 'val_split.json') )
eval_split = load_json_to_dic( os.path.join( os.path.dirname(__file__), 'eval_split.json') )


available_data = {
    'ycb': YCB_PATH,
    '3dnet': THREEDNET_PATH,
    'shapenet': SHAPENET_PATH
}

def load_dataset(name='ycb',
                 sampling='random',
                 partial=True):
    if name=='ycb':
        return UOPSIM(available_data[name], 
                    sampling=sampling,
                    partial=partial)
    else:
        return UOPSIM(available_data[name], val=eval_split[name],
                    sampling=sampling,
                    partial=partial)

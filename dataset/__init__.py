import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import os
from utils.file_utils import *
from dataset.uopsim import UOPSIM

YCB_PATH = '{}/ycb'
THREEDNET_PATH = '{}/3dnet'
SHAPENET_PATH = '{}/shapenet'

val_split = load_json_to_dic( os.path.join( os.path.dirname(__file__), 'val_split.json') )
eval_split = load_json_to_dic( os.path.join( os.path.dirname(__file__), 'eval_split.json') )

available_data = {
    'ycb': YCB_PATH,
    '3dnet': THREEDNET_PATH,
    'shapenet': SHAPENET_PATH
}

def load_dataset(root,
                 name='ycb',
                 sampling='random',
                 partial=True):
    if name=='ycb':
        return UOPSIM(root=available_data[name].format(root), 
                      sampling=sampling,
                      partial=partial)
    else:
        return UOPSIM(available_data[name].format(root), 
                      val=eval_split[name],
                      sampling=sampling,
                      partial=partial)
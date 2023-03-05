import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import os
import argparse
from itertools import product

from tqdm import tqdm

from dataset import load_dataset
from utils.file_utils import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', default='partial')
    parser.add_argument('--data', default='ycb')
    parser.add_argument('--sampling', default='random')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')
    parser.add_argument('--savedir', type=str, default='path to save sample data', help='dir to save')
    
    args = parser.parse_args()
    
    save_root = '{}/{}'.format(args.savedir, args.dtype)
    os.makedirs(save_root, exist_ok=True)

    trial = args.trial
    dir_name = "{}_{}".format(args.data, args.sampling)
    data_dir = os.path.join(save_root, dir_name)
    
    if args.dtype == 'partial':
        dataset = load_dataset(args.data, sampling=args.sampling, partial=True)
    elif args.dtype == 'whole':            
        dataset = load_dataset(args.data, sampling=args.sampling, partial=False)
    else:
        raise NotImplementedError
    
    for i in tqdm(range(len(dataset)), dir_name, disable=False):
        obj_name = dataset.object_names[i]
        for j in tqdm(range(trial), obj_name):
            save_dir = os.path.join(data_dir, obj_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.pkl".format(j))
            if os.path.isfile(save_path):
                continue
            data = dataset.__getitem__(i)
            obj_name = data['object_name']
            save_to_pickle(data, save_path)
            

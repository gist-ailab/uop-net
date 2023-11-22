import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import os
import argparse
from tqdm import tqdm

from dataset import load_dataset
from utils.file_utils import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='directory of uop data, end with ..../uop_data')
    parser.add_argument('--name', default='ycb', help='dataset name (ycb, 3dnet, shapenet)')
    parser.add_argument('--partial', action='store_true', help='partial or whole')
    parser.add_argument('--sampling', default='random')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')
    
    args = parser.parse_args()
    data_root = args.root
    trial = args.trial

    dataset = load_dataset(root=data_root, 
                           name=args.name, 
                           sampling=args.sampling, 
                           partial=args.partial)
    if args.partial:
        if os.path.isfile('hard_to_partial_sample.json'):
            hard_to_partial_sample = load_json_to_dic('hard_to_partial_sample.json')
        else:
            hard_to_partial_sample = {'ycb': [], '3dnet': [], 'shapenet': []}
    
    print(f"Start sampling {args.name} dataset")
    for i in range(len(dataset)):
        obj_name = dataset.object_names[i]
        if args.partial:
            partial_trial = 0
            partial_fail = 0
            error_rate = 0
            if obj_name in hard_to_partial_sample[args.name]:
                continue
            data_dir = os.path.join(data_root, args.name, obj_name, 'partial')
        else:
            data_dir = os.path.join(data_root, args.name, obj_name, 'whole')
        os.makedirs(data_dir, exist_ok=True)
        for j in range(trial):
            save_path = os.path.join(data_dir, f'{j}.pkl')
            if os.path.isfile(save_path):
                continue
            if args.partial:
                # try-except block is for the case that the partial data fails to be generated
                # for example, the object is too small to be sampled by capturing a point cloud
                print("object: {:<30} | Trial: {:<3}|{:<3} | Error rate: {:<4.2f}%".format(obj_name, j, trial, error_rate), end='\r')
                is_success = False
                while not is_success:
                    try:
                        partial_trial += 1
                        data = dataset.__getitem__(i)
                        is_success = True
                        save_to_pickle(data, save_path)
                    except:
                        partial_fail += 1
                        error_rate = partial_fail / partial_trial * 100
                        print("object: {:<30} | Trial: {:<3}|{:<3} | Error rate: {:<4.2f}%".format(obj_name, j, trial, error_rate), end='\r')
                        if error_rate > 50 and partial_trial > 10:
                            print("object: {:<30} | Trial: {:<3}|{:<3} | Error rate: {:<4.2f}% -> hard to sample".format(obj_name, j, trial, error_rate))
                            is_success = True
                            hard_to_partial_sample[args.name].append(obj_name)
                            save_dic_to_json(hard_to_partial_sample, 'hard_to_partial_sample.json') 
                            break
                        continue
            else:
                print(f"{obj_name} : {j}|{trial}", end='\r')
                data = dataset.__getitem__(i)
                save_to_pickle(data, save_path)
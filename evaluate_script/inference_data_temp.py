import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from itertools import product
from utils.file_utils import *
from tqdm import tqdm
import numpy as np 
from placement_module import *
import argparse
from multiprocessing import Process
from dataset import eval_split

def inference(target_obj_dir_list):
    for obj_dir in tqdm(target_obj_dir_list):
        for i in range(trial):
            save_dir = obj_dir.replace(data_root, eval_root)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.pkl".format(i))
            if os.path.isfile(save_path):
                try:
                    load_pickle(save_path)
                    continue
                except:
                    pass
            data_path = os.path.join(obj_dir, "{}.pkl".format(i))
            data = load_pickle(data_path)
            
            # input points
            points = data['points']
            # prediction mask
            if args.module == 'gt':
                preds = np.argmax(data['ins_labels'], axis=1)
                exp_result = place_module.get_stable_placement(points, preds)

            elif 'sop' in args.module:
                exp_result = place_module.get_stable_placement(points, gpu=True)
            else:
                exp_result = place_module.get_stable_placement(points)
            
            save_dir = obj_dir.replace(data_root, eval_root)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.pkl".format(i))
            save_to_pickle(exp_result, save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--module', default='sop', help='select placement module')
    
    parser.add_argument('--droot', type=str, help='dir to data')
    parser.add_argument('--evalto', type=str, help='eval dir')
    parser.add_argument('--dtype', default='partial')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')
    
    parser.add_argument('--maxprocess', type=int, default=10, help='max process')
    
    args = parser.parse_args()
    
    # dataset name
    dataset_list = ['3dnet', 'shapenet', 'ycb']
    sampling_method = ['random']
    
    data_root = os.path.join(args.droot, args.dtype)
    eval_root = os.path.join(args.evalto, args.dtype, args.module)
    
    place_module = load_placement_module(args.module)

    trial = args.trial
    for name, method in product(dataset_list, sampling_method):
        dir_name = "{}_{}".format(name, method)
        data_dir = os.path.join(data_root, dir_name)
        eval_list = eval_split[name]
        print("Load Data from {}\nSave Inference Result to {}".format(data_dir, eval_root))
        obj_dir_list = get_dir_list(data_dir)
        target_dir_list = [obj_dir for obj_dir in obj_dir_list if get_dir_name(obj_dir) in eval_list]
        obj_dir_list = target_dir_list
        # multi process
        if args.module == "sop" or args.module == "sop-whole":
            inference(obj_dir_list)
        
        else:
            ps_num = max(args.maxprocess-1, 1)
            step_size = len(obj_dir_list) // ps_num
            ps_list = [Process(target=inference, kwargs=({'target_obj_dir_list':obj_dir_list[step_size*i:step_size*(i+1)]})) for i in range(ps_num)]
            ps_list.append(Process(target=inference, kwargs=({'target_obj_dir_list':obj_dir_list[step_size*(ps_num):]})))
            for p in ps_list:
                p.start()
                # p.join()
            
            running = True
            while running:
                running = any([p.is_alive() for p in ps_list])
            for p in ps_list:
                p.close()
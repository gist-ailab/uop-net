import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils.calc_metric import calc_transform_diff
import pandas as pd
import argparse
import os
from itertools import product
from utils.file_utils import *
from tqdm import tqdm
import copy
import numpy as np
from dataset import eval_split

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--module', default='trimesh', help='select placement module')
    parser.add_argument('--droot', type=str, help='dir to data')
    parser.add_argument('--evalto', type=str, help='eval dir')

    parser.add_argument('--dtype', default='whole')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')
    
    args = parser.parse_args()
    
    # dataset name
    dataset_list = ['ycb', '3dnet', 'shapenet']
    sampling_method = ['random']
    
    temp = {}
    
    eval_root = os.path.join(args.evalto, args.dtype, args.module)
    
    trial = args.trial
    
    data_info = load_json_to_dic('{}_gt.json'.format(args.dtype))

    for name, method in product(dataset_list, sampling_method):
        dir_name = "{}_{}".format(name, method)
        eval_dir = os.path.join(eval_root, dir_name)
        temp.setdefault(name, {})
        data_metric = {}
        obj_metric = {
            "rotation": [],
            "transform_diff": [],
            "translation": [], 
            "l2norm": [],
            "infer": [],
        }
        
        eval_list = eval_split[name]
        
        for obj_dir in tqdm(get_dir_list(eval_dir), dir_name):
            obj_name = get_dir_name(obj_dir)
            if not obj_name in eval_list:
                continue
            data_metric.setdefault(obj_name, copy.deepcopy(obj_metric))
            target_metric = data_metric[obj_name]
            
            for i in range(trial):
                if not data_info[name][obj_name][i]:
                    continue
                eval_path = os.path.join(obj_dir, "{}.pkl".format(i))
                
                eval_result = load_pickle(eval_path)
                eval_info = eval_result['eval']
                
                # inference success or not
                if eval_info is None:
                    target_metric['infer'].append(False)

                else:
                    target_metric["infer"].append(True)
                    
                    temp_diff = {
                        "rotation": [],
                        "translation": [],
                        "l2norm": []
                    }
                    matrix_list = eval_info['matrix']
                    prev_mat = matrix_list[0]
                    for cur_mat in matrix_list[1:]:
                        diff = calc_transform_diff(prev_mat, cur_mat)
                        for k, v in diff.items():
                            temp_diff[k].append(v)
                        prev_mat = cur_mat
                    
                    # accumulated diff
                    for k, v in temp_diff.items():
                        target_metric[k].append(np.sum(v))
                    
                    # diff per step
                    target_metric['transform_diff'].append(temp_diff)
        # save to pkl
        save_path = eval_dir + ".pkl"
        save_to_pickle(data_metric, save_path)
        
        # save to csv per_obj
        save_path = eval_dir + "_per_obj.csv"
        # obj_name_list = [obj_name for obj_name in data_metric.keys()]
        obj_name_list = eval_list
        obj_name_list.sort()
        success_criterion = [i for i in range(1, 50)]

        rows = obj_name_list
        cols = ['rotation', 'translation', 'l2norm', 'infer']
        cols += ['success count_{}'.format(i) for i in success_criterion]
        data = np.zeros((len(rows), len(cols)))

        for obj_name in eval_list:
            if not any(data_info[name][obj_name]):
                continue

            metric = data_metric[obj_name]
            target_row = rows.index(obj_name)
            for target_col, col in enumerate(cols[:3]):
                v = metric[col]
                if len(v) > 0:
                    data[target_row][target_col] = sum(v) / len(v)
            
            data[target_row][3] = np.sum(metric['infer'])
            
            rotation_list = metric['rotation']
            for r in success_criterion:
                target_col = cols.index('success count_{}'.format(r))
                data[target_row][target_col] = sum(np.array(rotation_list) < r)
        
        df = pd.DataFrame(data, columns=cols, index=rows)
        df.to_csv(save_path)
        
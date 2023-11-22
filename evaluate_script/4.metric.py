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

result_msg = """
-----------------------------------------------------------------
Module           | {:<5} | {:<5} | {:<5} | {:<5}
rotation(deg)    | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
translation(cm)  | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
l2norm           | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
Success(<10deg)  | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
-----------------------------------------------------------------
"""

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', required=True, help='directory of uop data, end with ..../uop_data')
    parser.add_argument('--name', default='ycb', help='dataset name (ycb, 3dnet, shapenet)')
    parser.add_argument('--partial', action='store_true', help='partial or whole')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')

    parser.add_argument('--tilt', type=int, default=0, help='select placement module')
    args = parser.parse_args()

    # sampled data root
    data_root = os.path.join(args.root, args.name)
    args.partial = True

    # Placement Module
    module_list = {
        "UOP": "uop",
        "RPF": "ransac",
        "CHSA": "trimesh",
        "BBF": "primitive",
    }
    evaluation_result = {}
    for method, module_name in module_list.items():
        evaluation_result[method] = {}
        for obj_name in tqdm(eval_split[args.name], method):
            obj_dir = os.path.join(data_root, obj_name)
            data_dir = os.path.join(obj_dir, 'partial') if args.partial else os.path.join(obj_dir, 'whole')
            eval_dir = os.path.join(data_dir+'_eval', module_name)
            evaluation_result[method][obj_name] = {
                "rotation": [],
                "translation": [], 
                "l2norm": [],
                "infer": [],
            }
            for i in range(args.trial):
                # check evaluate result file -> inference fail or not
                eval_path = os.path.join(eval_dir, "tilt{}_{}.pkl".format(args.tilt, i))
                if not os.path.isfile(eval_path): # inference fail
                    evaluation_result[method][obj_name]['infer'].append(False)
                    continue
                evaluation_result[method][obj_name]['infer'].append(True)

                # load evaluate result
                eval_result = load_pickle(eval_path)
                eval_info = eval_result['eval']

                # calculate metric
                step_diff = {
                    "rotation": [],
                    "translation": [],
                    "l2norm": []
                }
                matrix_list = eval_info['matrix']
                prev_mat = matrix_list[0]
                for cur_mat in matrix_list[1:]:
                    diff = calc_transform_diff(prev_mat, cur_mat)
                    for k, v in diff.items():
                        step_diff[k].append(v)
                    prev_mat = cur_mat
                
                # accumulated diff
                for k, v in step_diff.items():
                    evaluation_result[method][obj_name][k].append(np.sum(v))

    # save to pkl
    if args.partial:
        save_path = os.path.join(args.root, "{}_partial_metric.pkl".format(args.name))
    else:
        save_path = os.path.join(args.root, "{}_whole_metric.pkl".format(args.name))
    save_to_pickle(evaluation_result, save_path)

    evaluation_result = load_pickle(save_path)

    compare_result = {}
    for method, module_name in module_list.items():
        # save as csv
        if args.partial:
            save_path = os.path.join(args.root, "{}_partial_metric_{}.csv".format(args.name, method))
        else:
            save_path = os.path.join(args.root, "{}_whole_metric_{}.csv".format(args.name, method))
        obj_name_list = eval_split[args.name]
        obj_name_list.sort()
        
        rows = obj_name_list
        cols = ["rotation", "translation", "l2norm", "Success Rate(<10deg)"]
        data = np.zeros((len(rows), len(cols)))

        for i, obj_name in enumerate(obj_name_list):
            for j, col in enumerate(cols[:3]):
                data[i, j] = np.mean(evaluation_result[method][obj_name][col])
            data[i, 3] = np.sum(np.array(evaluation_result[method][obj_name]['rotation']) < 10) / args.trial
        df = pd.DataFrame(data, columns=cols, index=rows)
        df.to_csv(save_path)
        compare_result[method] = {
            "rotation": np.mean(data[:, 0]),
            "translation": np.mean(data[:, 1]),
            "l2norm": np.mean(data[:, 2]),
            "success": np.mean(data[:, 3]),
        }

    # print result of each method
    print("Result of each method")
    print(result_msg.format(*module_list.keys(),
                            *[compare_result[method]['rotation'] for method in module_list.keys()],
                            *[compare_result[method]['translation']*100 for method in module_list.keys()],
                            *[compare_result[method]['l2norm'] for method in module_list.keys()],
                            *[compare_result[method]['success']*100 for method in module_list.keys()]))


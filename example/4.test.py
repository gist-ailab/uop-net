import argparse
import os
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname( os.path.dirname(os.path.realpath(__file__))))

from uop_sim.environment import EvaluateEnv
from utils.calc_metric import calc_transform_diff


result_msg = """
-----------------------------------------------------------------
Module           | {:<5} | {:<5} | {:<5} | {:<5}
rotation(deg)    | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
translation(cm)  | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
l2norm           | {:<5.2f} | {:<5.2f} | {:<5.2f} | {:<5.2f}
Success(<10deg)  | {} | {} | {} | {}
-----------------------------------------------------------------
"""

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='003_cracker_box')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    headless = not args.visualize
    env = EvaluateEnv(headless=headless)
    
    # Target Object
    uop_root = os.path.dirname( os.path.dirname( os.path.realpath(__file__)) )

    example_root = os.path.join(uop_root, 'example_results')
    sample_obj_dir = os.path.join(example_root, args.object)
    result_dir = os.path.join(sample_obj_dir, 'result')
    object_model = os.path.join(sample_obj_dir, "model.ttm")
    
    
    # Placement Module
    module_list = {
        "UOP": "uop",
        "RPF": "ransac",
        "CHSA": "trimesh",
        "BBF": "primitive",
    }
    results = {method: {} for method in module_list.keys()}
    for method, module_name in module_list.items():
        inference_result = torch.load(os.path.join(result_dir, "{}.pt".format(method)))
        rot = inference_result['rot']
        if rot is None:
            continue
        env.reset(object_model)
        eval_result = env.evaluate(rot)
        
        # inference success or not
        temp_diff = {
            "rotation": [],
            "translation": [],
            "l2norm": []
        }
        matrix_list = eval_result['matrix']
        prev_mat = matrix_list[0]
        for cur_mat in matrix_list[1:]:
            diff = calc_transform_diff(prev_mat, cur_mat)
            for k, v in diff.items():
                temp_diff[k].append(v)
            prev_mat = cur_mat
        
        # accumulated diff
        for k, v in temp_diff.items():
            results[method][k] = np.sum(v)
                    
    env.stop()
    
    
    # print result of each method 
    print("Result of each method")
    print(result_msg.format(*module_list.keys(),
                            *[results[method]['rotation'] for method in module_list.keys()],
                            *[results[method]['translation']*100 for method in module_list.keys()],
                            *[results[method]['rotation'] for method in module_list.keys()],
                            *[results[method]['rotation'] < 10 for method in module_list.keys()]))
    
    
    

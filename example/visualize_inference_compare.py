import os
import argparse

import sys
sys.path.append(os.path.dirname( os.path.dirname(os.path.realpath(__file__))))

from utils.file_utils import load_pickle
from utils.matplotlib_visualize_utils import visualize_exp_result, visualize_module_compare



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('partial_points', required=True, help='path to sampled partial points (endwith .pkl)')
    args = parser.parse_args()

    # Placement Module
    module_list = {
        "UOP": "uop",
        "RPF": "ransac",
        "CHSA": "trimesh",
        "BBF": "primitive",
    }

    target_obj_dir = os.path.dirname(os.path.dirname(args.partial_points))
    target_sampled_file = os.path.basename(args.partial_points)
    eval_dir = os.path.join(target_obj_dir, 'partial_eval')

    input_points = load_pickle(args.partial_points)['points']

    exp_results = {
        'input': {'points':input_points}
    }
    for method, module_name in module_list.items():
        exp_file = os.path.join(eval_dir, module_name, target_sampled_file)
        if not os.path.isfile(exp_file):
            print("Cannot find {} Module result".format(module_name))
            continue
        exp_result = load_pickle(exp_file)
        exp_results[method] = exp_result
    visualize_module_compare(exp_results)
        






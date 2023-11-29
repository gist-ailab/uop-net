import os
import argparse

import sys
sys.path.append(os.path.dirname( os.path.dirname(os.path.realpath(__file__))))

from utils.file_utils import load_pickle
from utils.matplotlib_visualize_utils import visualize_exp_result



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_file', required=True, help='path to experiment result file (endwith .pkl)')
    parser.add_argument('--module', required=True, help='module name (uop, ransac, trimesh, primitive)')
    args = parser.parse_args()

    exp_result = load_pickle(args.exp_file)
    visualize_exp_result(exp_result, args.module)


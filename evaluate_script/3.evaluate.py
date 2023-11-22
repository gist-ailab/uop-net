import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import os
from itertools import product
from utils.file_utils import *
from tqdm import tqdm
from placement_module import *
import argparse
from uop_sim.environment import EvaluateEnv
from multiprocessing import Process
from dataset import eval_split

def evaluate(target_obj_dir_list):
    env = EvaluateEnv(headless=True, tilt=int(args.tilt))
    for obj_dir in tqdm(target_obj_dir_list):
        data_dir = os.path.join(obj_dir, 'partial') if args.partial else os.path.join(obj_dir, 'whole')
        eval_dir = os.path.join(data_dir+'_eval', args.module)
        
        is_reset = False
        model_path = os.path.join(obj_dir, 'model.ttm')
        for i in range(args.trial):
            # check evaluate result
            eval_path = os.path.join(eval_dir, "tilt{}_{}.pkl".format(args.tilt, i))
            if os.path.isfile(eval_path):
                print("Already Evaluated in {}".format(eval_path))
                continue
            # load inference result
            infer_path = os.path.join(eval_dir, "{}.pkl".format(i))
            try:
                infer_result = load_pickle(infer_path)
            except:
                print("Inference Error in {}".format(infer_path))
                continue
            
            # check inference fail or not
            rot = infer_result['rot']
            if rot is None: # inference fail
                print("Inference Fail in {}".format(infer_path))
                continue
            
            # reset environment
            if not is_reset:
                env.reset(model_path)
                is_reset = True
            else:
                env.reset()

            infer_result['eval'] = env.evaluate(rot)
            save_to_pickle(infer_result, eval_path)
                    
    env.stop()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', required=True, help='directory of uop data, end with ..../uop_data')
    parser.add_argument('--name', default='ycb', help='dataset name (ycb, 3dnet, shapenet)')
    parser.add_argument('--partial', action='store_true', help='partial or whole')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')

    parser.add_argument('--module', default='uop', help='select placement module')
    parser.add_argument('--backbone', default=None, help='backbone of uop')
    parser.add_argument('--weight', default=None, help='uop weight path')

    parser.add_argument('--maxprocess', type=int, default=10, help='max process')
    
    parser.add_argument('--tilt', type=int, default=0, help='select placement module')
    
    args = parser.parse_args()
    
    # sampled data root
    data_root = os.path.join(args.root, args.name)

    # placement module
    module_kwargs = {}
    module_kwargs['module_name'] = args.module
    if args.backbone is not None:
        module_kwargs['backbone'] = args.backbone
        args.module = args.module + '-' + args.backbone
    if args.weight is not None:
        module_kwargs['weight'] = args.weight
        args.module = args.module + '-' + os.path.dirname(args.weight).split('/')[-1]

    eval_obj_dir_list = []
    for obj_name in eval_split[args.name]:
        obj_dir = os.path.join(data_root, obj_name)
        data_dir = os.path.join(obj_dir, 'partial') if args.partial else os.path.join(obj_dir, 'whole')
        eval_dir = os.path.join(data_dir+'_eval', args.module)
        if not os.path.isdir(eval_dir):
            print("No Inference Data in {}".format(eval_dir))
            continue
        if len(os.listdir(eval_dir)) < args.trial:
            print("Not Enough Inference Data in {}".format(eval_dir))
            continue
        eval_obj_dir_list.append(obj_dir)

    # multi process
    ps_num = max(args.maxprocess-1, 1)
    step_size = len(eval_obj_dir_list) // ps_num
    ps_list = [Process(target=evaluate, kwargs=({
        'target_obj_dir_list':eval_obj_dir_list[step_size*i:step_size*(i+1)]})) for i in range(ps_num)]
    ps_list.append(Process(target=evaluate, kwargs=({
        'target_obj_dir_list':eval_obj_dir_list[step_size*(ps_num):]})))
    for p in ps_list:
        p.start()
        # p.join()
    running = True
    while running:
        running = any([p.is_alive() for p in ps_list])
    for p in ps_list:
        p.close()
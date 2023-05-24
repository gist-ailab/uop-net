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
    
    for eval_dir in tqdm(target_obj_dir_list):
        obj_name = get_dir_name(eval_dir)
        is_reset = False
        model_path = os.path.join(args.droot ,name, obj_name, 'model.ttm')
        for i in range(trial):
            eval_path = os.path.join(eval_dir, "{}.pkl".format(i))
            try:
                eval_result = load_pickle(eval_path)
            except:
                print("load error")
                continue
            if eval_result['eval'] is not None:
                pass
            if not is_reset:
                env.reset(model_path)
                is_reset = True
            else:
                env.reset()
            rot = eval_result['rot']
            if rot is None:
                continue
            save_path = eval_path.replace(eval_root, save_root)
            if os.path.isfile(save_path):
                continue
            eval_result['eval'] = env.evaluate(rot)
            os.makedirs(str(Path(save_path).parent), exist_ok=True)
            save_to_pickle(eval_result, save_path)
                    
    env.stop()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--module', default='ransac', help='select placement module')
    
    parser.add_argument('--droot', type=str, help='dir to data')
    parser.add_argument('--evalto', type=str, help='eval dir')
    parser.add_argument('--dtype', default='partial')
    parser.add_argument('--trial', type=int, default=100, help='trial per object')
    parser.add_argument('--tilt', type=int, default=0, help='select placement module')
    parser.add_argument('--maxprocess', type=int, default=1, help='max process')
    
    args = parser.parse_args()
    
    # dataset name
    dataset_list = ['ycb', '3dnet', 'shapenet']
    sampling_method = ['random']
    
    data_root = os.path.join(args.droot, args.dtype)
    eval_root = os.path.join(args.evalto, args.dtype, args.module)
    
    if int(args.tilt) > 0:
        save_root = eval_root + "_tilt{}".format(args.tilt)
    else:
        save_root = eval_root
    os.makedirs(save_root, exist_ok=True)

    
    trial = args.trial
    for name, method in product(dataset_list, sampling_method):
        dir_name = "{}_{}".format(name, method)
        data_dir = os.path.join(eval_root, dir_name)

        obj_dir_list = get_dir_list(data_dir)
        eval_list = eval_split[name]
        eval_dir_list = []
        for obj_dir in obj_dir_list:
            obj_name = get_dir_name(obj_dir)
            if obj_name in eval_list:
                eval_dir_list.append(obj_dir)
        
        obj_dir_list = eval_dir_list
        # multi process
        
        ps_num = args.maxprocess
        step_size = len(obj_dir_list) // (ps_num - 1)
        ps_list = [Process(target=evaluate, kwargs=({'target_obj_dir_list':obj_dir_list[step_size*i:step_size*(i+1)]})) for i in range(ps_num)]
        ps_list.append(Process(target=evaluate, kwargs=({'target_obj_dir_list':obj_dir_list[step_size*(ps_num):]})))
        for p in ps_list:
            p.start()
            # p.join()
        
        running = True
        while running:
            running = any([p.is_alive() for p in ps_list])
        for p in ps_list:
            p.close()
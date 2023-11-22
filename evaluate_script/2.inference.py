import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils.file_utils import *
from tqdm import tqdm
from placement_module import *
import argparse
from multiprocessing import Process
from dataset import eval_split

def inference(module_kwargs, target_obj_dir_list):
    place_module = load_placement_module(**module_kwargs)
    for obj_dir in tqdm(target_obj_dir_list):
        data_dir = os.path.join(obj_dir, 'partial') if args.partial else os.path.join(obj_dir, 'whole')
        eval_dir = os.path.join(data_dir+'_eval', args.module)
        os.makedirs(eval_dir, exist_ok=True)
        for i in range(args.trial):
            data_path = os.path.join(data_dir, "{}.pkl".format(i))
            eval_path = os.path.join(eval_dir, "{}.pkl".format(i))
            if os.path.isfile(eval_path):
                continue
            data = load_pickle(data_path)
            # input points
            points = data['points']

            # prediction mask
            exp_result = place_module.get_stable_placement(points)
            save_to_pickle(exp_result, eval_path)

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
        args.module = args.module + '-' + os.path.dirname(args.weight).split('/')[-1] # get name from dir path of weight

    eval_obj_dir_list = []
    for obj_name in eval_split[args.name]:
        obj_dir = os.path.join(data_root, obj_name)
        data_dir = os.path.join(obj_dir, 'partial') if args.partial else os.path.join(obj_dir, 'whole')
        if not os.path.isdir(data_dir):
            print("No Sampled Data in {}".format(data_dir))
            continue
        if len(os.listdir(data_dir)) < args.trial:
            print("Not Enough Sampled Data in {}".format(data_dir))
            continue
        eval_obj_dir_list.append(obj_dir)

    # multi process
    if 'uop' in args.module or args.module == "ransac":
        inference(module_kwargs, eval_obj_dir_list)
    else:
        ps_num = max(args.maxprocess-1, 1)
        step_size = len(eval_obj_dir_list) // ps_num
        ps_list = [Process(target=inference, kwargs=({
            'module_kwargs': module_kwargs,
            'target_obj_dir_list':eval_obj_dir_list[step_size*i:step_size*(i+1)]})) for i in range(ps_num)]
        ps_list.append(Process(target=inference, kwargs=({
            'module_kwargs': module_kwargs,
            'target_obj_dir_list':eval_obj_dir_list[step_size*(ps_num):]})))
        for p in ps_list:
            p.start()
            p.join()
        
        running = True
        while running:
            running = any([p.is_alive() for p in ps_list])
        for p in ps_list:
            p.close()
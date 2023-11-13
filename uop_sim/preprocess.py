import os
import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path().absolute()))

from natsort import natsorted
from tqdm import tqdm
from utils.file_utils import *
import copy
import numpy as np

import trimesh
from pyrep import PyRep

from utils.pyrep_utils import convert_mesh_to_scene_object 
from utils.open3d_utils import sampling_points_from_mesh, save_point_cloud
from utils.file_utils import *
import argparse

MANIFOLD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'thirdparty/Manifold/build')

class DataProcessing:
    @staticmethod
    def check_meshext(file_name):
        if os.path.splitext(file_name)[1].lower() in (".ply", ".dae", ".obj"):
            return True
        else:
            return False

    @staticmethod
    def convert_naming_rule(name):
        name = name.replace(' ', '_')
        name = name.replace('-', '_')

        return name

    @staticmethod
    def get_mesh(cfg, split=None):
        meta_path = cfg['save_root'] + '_meta.json'
        if os.path.isfile(meta_path):
            data_info = DataProcessing.load_meta(cfg, meta_path)

        else:
            if cfg['data_type']=='ycb':
                data_info = DataProcessing.get_YCB_mesh(cfg['data_root'])
            elif cfg['data_type']=='ycb-texture':
                data_info = DataProcessing.get_YCB_mesh(cfg['data_root'], texture=True)
            elif cfg['data_type']=='shapenet':
                data_info = DataProcessing.get_ShapeNet_mesh(cfg['data_root'])
            elif cfg['data_type']=='3dnet':
                data_info = DataProcessing.get_3DNet_mesh(cfg['data_root'])
            else:
                raise NotImplementedError        
            DataProcessing.save_meta(cfg, data_info)

        return DataProcessing.split_data(data_info, split)
            
        
    @staticmethod
    def load_meta(cfg, meta_path):
        meta_info = load_json_to_dic(meta_path)
        for k, v in meta_info.items():
            meta_info[k][1] = cfg['data_root'] + v[1]
        
        return meta_info
    
    @staticmethod
    def save_meta(cfg, meta_info):
        temp_info = copy.deepcopy(meta_info)
        meta_path = cfg['save_root'] + '_meta.json'
        for k, v in temp_info.items():
            temp_info[k] = [v[0], v[1].replace(cfg['data_root'], '')]
        save_dic_to_json(temp_info, meta_path)

    @staticmethod
    def split_data(data_info, target=None):
        temp_data = {}
        for k, v in data_info.items():
            if target is None:
                temp_data[k] = v[1]
            elif v[0] in target:
                temp_data[k] = v[1]
            else:
                continue
        return temp_data

    @staticmethod
    def get_YCB_mesh(data_root, texture=False):
        """get YCB raw mesh from data_root

        Args:
                data_root (str): [ycb dataset root]
                looks like >>>
                ycb-tools/models/ycb # data root
                ├── 001_chips_can
                ├── 002_master_chef_can
                ├── ...
                ├── 076_timer
                └── 077_rubiks_cube
                
                texture (bool): get ycb mesh with texture(*.dae)
                *but cannot convert to point cloud
        
        Returns:
                meta_info (dict): dict(object name, path to mesh file) for All google_16k mesh
                
        """
        data_info = {}
        ycb_list = [join(data_root, p) for p in os.listdir(data_root)]
        ycb_sorted_list = natsorted(ycb_list)
        for target_ycb_folder in tqdm(ycb_sorted_list):
            target_name = target_ycb_folder.split('/')[-1]
            target_name = target_name.replace("-", "_")
            
            # google_16k has high quality
            if "google_16k" in os.listdir(target_ycb_folder):
                pass
            else:
                continue
            if not texture:
                mesh_file = os.path.join(target_ycb_folder, "google_16k", "nontextured.ply")
            else:
                mesh_file = os.path.join(target_ycb_folder, "google_16k", "textured.dae")
            data_info[target_name] = mesh_file
        
        meta_info = {}
        meta_idx = 0
        for target_name, mesh_file in data_info.items():
            meta_info[target_name] = [meta_idx, mesh_file]
            meta_idx += 1
        
        return meta_info

    @staticmethod
    def get_ShapeNet_mesh(data_root):
        """get shapenet raw mesh from data_root

        Args:
                data_root (str): [shapenet dataset root]
                looks like >>>
                ShapeNetCore.v1 # data root
                ├── 02691156
                ├── 02691156.csv
                ├── 02691156.zip
                ├── ...
                ├── count-models.sh
                ├── get-metadata.sh
                ├── jq
                ├── README.txt
                └── taxonomy.json
        Returns:
                meta_info (dict): dict(object name, path to mesh file) for All mesh
                
        """

        taxonomy = load_json_to_dic(os.path.join(data_root, 'taxonomy.json')) # list 
        
        class_id_to_name = {}
        data_info = {}

        for class_info in taxonomy:
            class_id = class_info['synsetId']
            class_name = class_info['name'].split(',')[0]
            class_name = DataProcessing.convert_naming_rule(class_name)
            class_id_to_name[class_id] = class_name

        class_dir_list = get_dir_list(data_root)
        class_dir_sorted_list = natsorted(class_dir_list)
        for class_dir in tqdm(class_dir_sorted_list):
            class_id = get_dir_name(class_dir)
            class_dir = join(data_root, class_dir)
            obj_dir_list = get_dir_list(class_dir)
            obj_dir_sorted_list = natsorted(obj_dir_list)
            for idx, obj_dir in enumerate(obj_dir_sorted_list):
                mesh_path = os.path.join(obj_dir, 'model.obj')
                if os.path.isfile(mesh_path.replace('.obj', '.mtl')):
                    pass
                else:
                    continue
                mesh_name = "{}_{:03d}".format(class_id_to_name[class_id], idx)
                data_info[mesh_name] = mesh_path
        
        meta_info = {}
        meta_idx = 0
        for target_name, mesh_file in data_info.items():
            meta_info[target_name] = [meta_idx, mesh_file]
            meta_idx += 1
        
        return meta_info
    
    @staticmethod
    def get_3DNet_mesh(data_root):
        """get 3DNet raw mesh from data_root

        Args:
                data_root (str): [3DNet dataset root]
                looks like >>>
                    3DNet_raw/ # data_root
                    ├── Cat10_ModelDatabase
                    ├── Cat10_TestDatabase
                    ├── Cat200_ModelDatabase
                    ├── Cat60_ModelDatabase
                    └── Cat60_TestDatabase
        Returns:
                meta_info (dict): dict(object name, path to mesh file) for All mesh
                
        """
        data_info = {}
        type_list = [join(data_root, p) for p in os.listdir(data_root)]
        for type_dir in tqdm(type_list):
            if "Test" in type_dir:
                continue
            cat_list = [join(type_dir, p) for p in os.listdir(type_dir)]
            
            for cat_dir in cat_list:
                target_name = cat_dir.split('/')[-1]
                target_name = target_name.replace("-", "_")
                for mesh_file in [join(cat_dir, p) for p in os.listdir(cat_dir)]:
                    if DataProcessing.check_meshext(mesh_file):
                        data_info.setdefault(target_name, [])
                        data_info[target_name].append(mesh_file)

        meta_info = {}
        meta_idx = 0
        for target_name, v in data_info.items():
            for idx, mesh_file in enumerate(v):
                meta_info["{}_{:03d}".format(target_name, idx)] = [meta_idx, mesh_file]
                meta_idx += 1
                
        return meta_info

def normalize_point_cloud(points : np.ndarray) -> np.ndarray:
    assert points.shape[0] > 0
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_value = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_value
    
    return points, centroid, max_value


def convert_to_watertight(input_mesh, output_mesh):
    print(f'watertight running on "{input_mesh}"')
    #!TODO: This custom path should be modified after sudo make install
    cmd = f'{MANIFOLD_PATH}/manifold {input_mesh} {output_mesh}'
    retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    if retobj.returncode != 0:
        print(f'manifold failed on "f{input_mesh}"')
        if retobj.stdout != '': print(f'{retobj.stdout}')
        if retobj.stderr != '': print(f'{retobj.stderr}')


def preprocess_mesh(data_root, save_root):
    data_type = os.path.basename(save_root)
    
    
    if data_type == "shapenet":
        mesh_info = DataProcessing.get_ShapeNet_mesh(os.path.join(data_root, "ShapeNetCore.v1"))
    elif data_type == "3DNet":
        mesh_info = DataProcessing.get_3DNet_mesh(os.path.join(data_root, "3DNet_raw"))
    elif data_type == "ycb":
        mesh_info = DataProcessing.get_YCB_mesh(os.path.join(data_root, "models/ycb"))
    elif data_type == "ycb-texture":
        mesh_info = DataProcessing.get_YCB_mesh(os.path.join(data_root, "models/ycb"), texture=True)
        
    for obj_name, v in mesh_info.items():
        mesh_file = v[1]
        obj_dir = os.path.join(save_root, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        target_file = os.path.join(obj_dir, "mesh_watertight.ply")
        if os.path.isfile(target_file):
            continue

        # raw mesh
        mesh = trimesh.load_mesh(mesh_file)
        mesh.export(os.path.join(obj_dir, "mesh_raw.ply"))
        
        # normalize mesh file
        vertex = mesh.vertices
        vertex, _, _ = normalize_point_cloud(vertex)
        mesh.vertices = vertex
        mesh.export(os.path.join(obj_dir, "mesh_normalize.ply"))
        mesh.export(os.path.join(obj_dir, "mesh_normalize.obj"))
        
        # for physics
        # watertight mesh file
        input_mesh = os.path.join(obj_dir, "mesh_normalize.obj")
        output_mesh = os.path.join(obj_dir, "mesh_watertight.obj")
        if mesh.is_watertight:
            shutil.copy(input_mesh, output_mesh)
        else:
            convert_to_watertight(input_mesh, output_mesh)
        mesh = trimesh.load_mesh(output_mesh)
        mesh.export(os.path.join(obj_dir, "mesh_watertight.ply"))


    os.system("rm {}/*/*.obj".format(save_root))
    os.system("rm {}/*/*.mtl".format(save_root))
    os.system("rm {}/*/*.png".format(save_root))
    os.system("rm {}/*/mesh_raw.ply".format(save_root))
    os.system("rm {}/*/mesh_normalize.ply".format(save_root))

def preprocess_scene_model(data_root):
    pr = PyRep()
    pr.launch(headless=True)
    pr.start()
    
    for obj_dir in tqdm([os.path.join(data_root, p) for p in os.listdir(data_root)]):
        obj_name = get_dir_name(obj_dir)
        mesh_file = os.path.join(obj_dir, "mesh_watertight.ply")
        save_path = os.path.join(obj_dir, "model.ttm")
        if not os.path.isfile(mesh_file):
            continue
        if os.path.isfile(save_path):
            continue
        convert_mesh_to_scene_object(pr, mesh_file, save_path, 
                                     bbox_size=0.2, target_name=obj_name)

def preprocess_pointcloud(data_root):
    for obj_dir in tqdm([os.path.join(data_root, p) for p in os.listdir(data_root)]):
        mesh_file = os.path.join(obj_dir, "mesh_watertight.ply")
        target_file = os.path.join(obj_dir, "point_cloud.ply")
        if not os.path.isfile(mesh_file):
            continue
        if os.path.isfile(target_file):
            continue
        pcd = sampling_points_from_mesh(mesh_file=mesh_file,
                                      number_of_points=10000,
                                      method='uniform-poisson')
        save_path = os.path.join(obj_dir, "point_cloud.ply")
        save_point_cloud(pcd, save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/raeyo/workspace/uop-net')
    parser.add_argument('--data_type', default='ycb') # ycb, shapenet, 3dnet, ycb-texture
    args = parser.parse_args()
    
    save_root = os.path.join(args.root , 'uop_data', args.data_type)
    os.makedirs(save_root, exist_ok=True)
    
    # check 3d model watertight
    preprocess_mesh(args.root, save_root)
    
    # sampling points from 3d model(mesh)
    preprocess_pointcloud(save_root)
    
    # generate simulation scene model
    preprocess_scene_model(save_root)


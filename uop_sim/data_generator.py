import os
import sys
import argparse
import copy
import time
from time import ctime
from itertools import product
from logging import warning

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.file_utils import *
from environment import PlacementEnv, InspectionEnv
from labeling import clustering_sampled_pose, get_instance_label_from_clustered_info


data_file = {
    # preprocess
    "mesh": "mesh_watertight.ply",
    "pointcloud": "point_cloud.ply",
    "model": "model.ttm",
    
    # sampling
    "stability": "stability.pkl",
    "cluster": "cluster.pkl",
    
    # labeling
    "label": "label.pkl",
    
    # after inspection
    "inspect_cluster": "cluster_inspected.pkl",
    "inspect_label": "label_inspected.pkl"
}


class UOPDataGenerator():
    def __init__(self, cfg):
        self.logger = get_logger("{} Data Generator".format(cfg['data_type']))
        self.headless = cfg['headless']
        
        # Sampling Stable Pose
        self.orientation_grid_size = cfg['orientation_grid_size'] # grid for each rotation
        self.orientation_list = [
            (x, y, z) for x, y, z in product(np.linspace(0, 2*np.pi, self.orientation_grid_size),
                                             np.linspace(0, 2*np.pi, self.orientation_grid_size),
                                             np.linspace(0, 2*np.pi, self.orientation_grid_size))
        ]

        # simulation
        self.time_step = cfg['time_step'] # 5ms
        self.tolerance = cfg['tolerance']
        self.ins_tolerance = cfg['inspection_tolerance']
        self.max_step = cfg['max_step']
        self.tilt = cfg['tilt']
        self.min_step = cfg['min_step']
        self.table_grid_size = cfg['table_grid_size']

        self.env = PlacementEnv(grid_size=self.table_grid_size,
                                headless=self.headless,
                                time_step=self.time_step,
                                tolerance=self.tolerance,
                                max_step=self.max_step)
        
    def convert_to_sampling_env(self):
        self.env.stop()
        self.env = PlacementEnv(grid_size=self.table_grid_size,
                                headless=self.headless,
                                time_step=self.time_step,
                                tolerance=self.tolerance,
                                max_step=self.max_step)
    
    def convert_to_labeling_env(self):
        self.env.stop()
        self.env = InspectionEnv(headless=self.headless,
                                 time_step=self.time_step,
                                 tilt=self.tilt,
                                 tolerance=self.ins_tolerance,
                                 min_step=self.min_step)
            
    def _simulate_stability(self, model_path, save_path):
        # for initialize
        assert isinstance(self.env, PlacementEnv), "please inspection to False"

        try:
            save_to_pickle({}, save_path)
            
            stability = self.env.get_stability(model_path, self.orientation_list)
            
            save_to_pickle(stability, save_path)
        except:
            raise Exception("Simulate stability error")

    def _clustering_stability(self, stability_path, save_path):
        try:
            clustered_info = clustering_sampled_pose(stability_path)
            save_to_pickle(clustered_info, save_path)
        except:
            raise Exception("Clustering stability error")
    
    def _labeling_stability(self, pc_path, clustered_info, save_path):
        try:
            label = get_instance_label_from_clustered_info(pc_path, clustered_info)
            save_to_pickle(label, save_path)
        except:
            raise Exception("Labeling stability error")
    
    def _inspecting_cluster(self, model_path, clustered_info, save_path):
        assert isinstance(self.env, InspectionEnv), "please convert env"
        try:
            clustered_info = self.env.inspect_clustered_info(model_path, clustered_info)
            save_to_pickle(clustered_info, save_path)
        except:
            raise Exception("Inspecting clustering error")
    
    def preprocess(self, object_name, mesh_file, save_dir):
        assert os.path.isfile(mesh_file), "[File Path Error] No file at {}".format(mesh_file)
        print(">>> Preprocess UOP Data >>> {}\n{:<35} {}".format(ctime(time.time()),"Target file:", object_name))
        
        '''1. Get raw mesh file'''
        print("{:<35} ...".format('Get raw mesh file:'), end='\r')
        _, ext =os.path.splitext(mesh_file)
        mesh_path = join(save_dir, "mesh{}".format(ext))
        if os.path.isfile(mesh_path):
            self.logger.debug("\n>>> Already copied raw mesh file at {}".format(mesh_path))
        else:
            copy_file(mesh_file, mesh_path)
            self.logger.debug("\n>>> Copy raw mesh file {}\t>> {}".format(mesh_file, mesh_path))
        print("{:<35} OK!".format('Get raw mesh file:'))
        
        
        '''2. Convert to CoppeliaSim Model'''
        print("{:<35} ...".format('Convert to CoppeliaSim Model:'), end='\r')
        model_path = join(save_dir, "model.ttm")
        if os.path.isfile(model_path):
            self.logger.debug("\n>>> Already convert CoppeliaSim Model {}".format(model_path))
        else:
            self.logger.debug("\n>>> Convert to CoppeliaSim Model {}\t>> {} ...".format(mesh_path, model_path))
            mesh_to_pc_scale = self._to_pyrep_model(mesh_file=mesh_file,        # TODO: is used?
                                                    save_path=model_path,
                                                    target_name=object_name)
        print("{:<35} OK!".format('Convert to CoppeliaSim Model:'))
        
        '''2. Convert to Point Cloud'''
        print("{:<35} ...".format('Convert to Point Cloud:'), end='\r')
        pc_path = join(save_dir, "point_cloud.ply")
        if os.path.isfile(pc_path):
            self.logger.debug("\n>>> Already sampling Point Cloud {}".format(pc_path))
        else:
            self.logger.debug("\n>>> Sampling the Point Cloud {}\t>> {} ...".format(mesh_file, pc_path))
            self._to_point_cloud(mesh_file=mesh_path,
                                 save_path=pc_path)
        print("{:<35} OK!".format('Convert to Point Cloud:'))
        
    def sampling(self, object_name, save_dir, force_run=False):
        # check env
        if isinstance(self.env, InspectionEnv):
            warning("Please use proper env")
            self.convert_to_sampling_env()
        
        print(">>> Sampling UOP Data >>> {}\n{:<35} {}".format(ctime(time.time()), "Target file:", object_name))
        
        model_path = join(save_dir, "model.ttm")
        assert os.path.isfile(model_path), "No model file in {}".format(save_dir)
        
        '''2. Sampling the Stable Pose'''
        print("{:<35} ...".format('Simulate Stability of Pose:'), end='\r')
        stability_path = join(save_dir, data_file["stability"])
        if os.path.isfile(stability_path) and not force_run:
            self.stability = load_pickle(stability_path)
            if self.stability == {}:
                self._simulate_stability(model_path, stability_path)
                self.stability = load_pickle(stability_path)
            else:
                self.logger.debug("\n>>> Already calculate stability {}".format(stability_path))
        else:
            self.logger.debug("\n>>> Simulate stability to {} ...".format(stability_path))
            self._simulate_stability(model_path, stability_path)

        print("{:<35} OK!".format('Simulate Stability of Pose:'))

    def labeling(self, object_name, save_dir, inspection=False):
        if inspection:
            if isinstance(self.env, PlacementEnv):
                warning("Please use proper env")
                self.convert_to_labeling_env()
                
        print(">>> Labeling UOP Data >>> {}\n{:<35} {}".format(ctime(time.time()), "Target file:", object_name))
        
        '''Check File Existancy'''
        model_path = join(save_dir, data_file["model"])
        pc_path = join(save_dir, data_file["pointcloud"])
        stability_path = join(save_dir, data_file["stability"])
        assert os.path.isfile(pc_path), "No point cloud file in {}".format(save_dir)
        assert os.path.isfile(stability_path), "No stability file in {}".format(save_dir)
        
        '''Clustering Sampled Stable Pose'''
        print("{:<35} ...".format('Clustering Stability of Object:'), end='\r')
        cluster_path = join(save_dir, data_file["cluster"])
        if os.path.isfile(cluster_path):
            self.logger.debug("\n>>> Already clustering stability {}".format(cluster_path))
        else:
            self.logger.debug("\n>>> Clustering Stability to {} ...".format(stability_path))
            self._clustering_stability(stability_path, cluster_path)
        clustered_info = load_pickle(cluster_path)
        print("{:<35} OK!".format('Clustering Stability of Object:'))
        
        '''Labeling Clustered Stable Pose to Point Cloud'''
        print("{:<35} ...".format('Labeling Stability of Object:'), end='\r')
        label_path = join(save_dir, data_file["label"])
        if os.path.isfile(label_path):
            self.logger.debug("\n>>> Already labeling stability {}".format(label_path))
        else:
            self.logger.debug("\n>>> Labeling Stability to {} ...".format(stability_path))
            self._labeling_stability(pc_path, clustered_info, label_path)
        print("{:<35} OK!".format('Labeling Stability of Object:'))

        '''Inspectig Cluster and Label Stable Pose to Point Cloud'''
        if inspection:
            print("{:<35} ...".format('Inspecting Clustered Info:'), end='\r')
            inspect_cluster_path = join(save_dir, data_file["inspect_cluster"])
            if os.path.isfile(inspect_cluster_path):
                self.logger.debug("\n>>> Already Inspect Clustering Info {}".format(cluster_path))
            else:
                self.logger.debug("\n>>> Inspecting Clustered Info to {} ...".format(stability_path))
                self._inspecting_cluster(model_path, clustered_info, inspect_cluster_path)
            clustered_info = load_pickle(inspect_cluster_path)
            print("{:<35} OK!".format('Clustering Stability of Object:'))

            print("{:<35} ...".format('Qualifying Stability of Object:'), end='\r')
            label_path = join(save_dir, data_file["inspect_label"])
            if os.path.isfile(label_path):
                self.logger.debug("\n>>> Already Qualifying stability {}".format(label_path))
            else:
                self.logger.debug("\n>>> Qualifying Stability to {} ...".format(stability_path))
                self._labeling_stability(pc_path, clustered_info, label_path)
            print("{:<35} OK!".format('Qualifying Stability of Object:'))

    def stop(self):
        self.env.stop()


class GenerateManager():
    log_format = {
        'sampling': {},
        'labeling': {},
    }
    
    def __init__(self, cfg, split_idx):
        self.cfg = cfg

        self.pid = split_idx
        self.total = self.load_total_log()
        self.success = self.load_success_log()
        self.generator = UOPDataGenerator(cfg)
    
    @staticmethod
    def check_under(object_name, under):
        if under < 0:
            return True
        idx = -1
        while object_name[idx].isnumeric():
            idx -= 1
        if idx == -1: # no numeric idx
            return True 
        else:
            if int(object_name[idx+1:]) >= under:
                return False
            else:
                return True
        
    def load_total_log(self):
        log = copy.deepcopy(self.log_format)
        temp_file = self.cfg['save_root'] + '_total_{}.json'.format(self.pid)
        if os.path.isfile(temp_file):
            log = load_json_to_dic(temp_file)
        else:
            log_file = self.cfg['save_root'] + '_total.json'
            if not os.path.isfile(log_file):
                save_dic_to_json(log, log_file)
            else:
                log = load_json_to_dic(log_file)
            
        return log
    
    def save_total_log(self, final=False):
        log_file = self.cfg['save_root'] + '_total_{}.json'.format(self.pid)
        if final:
            if os.path.isfile(log_file):
                os.remove(log_file)
            log_file = self.cfg['save_root'] + '_total.json'
            if os.path.isfile(log_file):
                log = load_json_to_dic(log_file)
                for k in log.keys():
                    self.total[k].update(log[k])
            
        save_dic_to_json(self.total, log_file)
    
    def load_success_log(self):
        log = copy.deepcopy(self.log_format)
        temp_file = self.cfg['save_root'] + '_success_{}.json'.format(self.pid)
        if os.path.isfile(temp_file):
            log = load_json_to_dic(temp_file)
        else:
            log_file = self.cfg['save_root'] + '_success.json'
            if not os.path.isfile(log_file):
                save_dic_to_json(log, log_file)
            else:
                log = load_json_to_dic(log_file)
        return log
    
    def save_success_log(self, final=False):
        log_file = self.cfg['save_root'] + '_success_{}.json'.format(self.pid)
        if final:
            if os.path.isfile(log_file):
                os.remove(log_file)
            log_file = self.cfg['save_root'] + '_success.json'
            if os.path.isfile(log_file):
                log = load_json_to_dic(log_file)
                for k in log.keys():
                    self.success[k].update(log[k])
        save_dic_to_json(self.success, log_file)
    
    def sampling(self, object_name):
        if not self.check_under(object_name, self.cfg['under']):
            print("Pass upper {}: {}".format(self.cfg['under'], object_name))
            return
        save_dir = join(self.cfg['save_root'], object_name)
        model_file = join(save_dir, "model.ttm")
        if not os.path.isfile(model_file):
            print("Please preprocessing {} first!".format(object_name))
            return
        if object_name in self.total['sampling'].keys():
            return
        
        self.total['sampling'][object_name] = True
        self.save_total_log()
        
        try:
            self.generator.sampling(object_name, save_dir)
            self.success['sampling'][object_name] = True
            self.save_success_log()
        except Exception as e:
            error_msg = "Object Name: {:<20} |Error msg: {}".format(object_name, e)
            print(error_msg)
        
    def labeling(self, object_name):
        if not self.check_under(object_name, self.cfg['under']):
            print("Pass upper {}: {}".format(self.cfg['under'], object_name))
            return
        save_dir = join(self.cfg['save_root'], object_name)
        if not object_name in self.success['sampling'].keys():
            print("Please sampling {} first!".format(object_name))
            return
        
        if object_name in self.total['labeling'].keys():
            return

        self.total['labeling'][object_name] = True
        self.save_total_log()

        try:
            self.generator.labeling(object_name, save_dir, inspection=self.cfg['inspection'])
            self.success['labeling'][object_name] = True
            self.save_success_log()
        except Exception as e:
            error_msg = "Object Name: {:<20} |Error msg: {}".format(object_name, e)
            print(error_msg)
    
    def convert_env(self):
        if self.cfg['inspection']:
            self.generator.convert_to_labeling_env()
    
    def stop(self):
        self.generator.stop()
        self.save_success_log(final=True)
        self.save_total_log(final=True)
        
        print("*****Data Generate Result*****")
        print("Data: {}".format(cfg['data_type']))
        print("UOP Data: {}".format(cfg['save_root']))
        
        for process_key in self.log_format.keys():
            print("{} | {} >>> {}".format(process_key, len(self.total[process_key]), len(self.success[process_key])))
            fail_case = set(self.total[process_key]) - set(self.success[process_key])
            print("FAIL_CASE: {}".format(fail_case))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data_type', type=str,
                        default='ycb') # ycb, shapenet, 3dnet, ycb-texture
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--under', type=int, 
                        default=-1)
    parser.add_argument('--split', type=int, 
                        default=1)
    
    args = parser.parse_args()
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.yaml')
    cfg = load_yaml_to_dic(config_path)

    cfg['data_type'] = args.data_type
    cfg['headless'] = not args.vis
    cfg['inspection'] = args.inspect
    cfg['under'] = args.under
    cfg['save_root'] = os.path.join(args.root, 'uop_data', cfg['data_type'])
    
    manager = GenerateManager(cfg, args.split)
    obj_list = os.listdir(cfg['save_root'])
    obj_list.sort()

    print(f">> Start sampling ...")
    for object_name in obj_list:
        manager.sampling(object_name)
    print(f">> End sampling!")

    manager.convert_env() # convert to inspection env
    
    print(f">> Start labeling ...")
    for object_name in obj_list:
        manager.labeling(object_name)
    print(f">> End labeling!")

    manager.stop()

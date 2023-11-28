import open3d as o3d
import os
from utils.file_utils import *
from utils.open3d_utils import sampling_points_from_mesh

from utils.matplotlib_visualize_utils import *
from dataset import UOPSIM
import numpy as np
import random
from utils.capture_points_from_mesh import MeshCapture
from placement_module import load_placement_module
import torch

""" Visualize partial view and each modules' results

for each partial view (1 ~ 1000)

1. visualize whole points + parital points + camera view (1 ~ 1000) -> open3d
2. visualize partial points + each module results (1 ~ 1000) -> matplotlib

- partial points (input points)

- RPF result 
- CHSA result
- BBF result
- UOP result

1. random 100 view , metric with 100 trial
-> rotation, translation, l2norm (/infer) -> uop is over best 
-> success rate (/trial) -> uop is best
2. 100 view and all sample has placement label, metric with 100 trial
-> ??
-> uop is best?

3. 100 random view, metric with inferenced trial
-> rotation, translation, l2norm (/infer) -> uop is over best
-> success rate (/infer) -> uop is over best



"""




class PointCloudVisualizer:
    """Reference:
    http://www.open3d.org/docs/0.9.0/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer
    """
    class State:
        initalize = 0
        
        partial_view_sampling = 1

        inference = 2
        
        end = 3


    def __init__(self, uopsim_obj_dir):
        self.obj_dir = uopsim_obj_dir

        self.mesh_file = os.path.join(uopsim_obj_dir, "mesh_watertight.ply")
        self.cluster = load_pickle(os.path.join(uopsim_obj_dir, "inspected_zaxis.pkl"))
        
        self.whole_pcd = sampling_points_from_mesh(self.mesh_file, 30000)
        self.whole_points = np.asarray(self.whole_pcd.points, dtype=np.float32)
        self.whole_pcd.colors = o3d.utility.Vector3dVector(np.zeros((self.whole_points.shape[0], 3))+0.5)
        
        #TODO:
        self.capture_tool = MeshCapture()
        self.param_list = [os.path.join(self.capture_tool.param_dir, f"{i}.json") for i in range(1, 1001)]


        # Placement Module
        self.module_list = {
            "UOP": load_placement_module("uop"),
            "RPF": load_placement_module("ransac"),
            "CHSA": load_placement_module("trimesh"),
            "BBF": load_placement_module("primitive"),
        }

        # initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.state = self.State.initalize
        self.init_param = None
        self.idx = 0
        self.total_num = 999
        
        # # add current PointCloud to window
        self.vis.add_geometry(self.whole_pcd)
        
        # load render option from file : RenderOption
        self.render_option = self.vis.get_render_option()
        self.render_option.load_from_json('render_option.json')
        
        
        # register callback function 
        self.vis.register_animation_callback(self.ani_callback)
        self.vis.register_key_callback(ord("I"), self.init_viewpoint)

        # activate window
        self.vis.run()
    
    @staticmethod #TODO
    def create_camera(intrinsic, extrinsic, scale=0.1):
        return o3d.geometry.LineSet.create_camera_visualization(intrinsic, extrinsic, scale)

    def init_viewpoint(self, vis):
        self.init_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        self.state = self.State.partial_view_sampling

        self.idx = 0
        self.partial_points = []
        self.frame_param = []

    
    def reset_viewpoint(self):
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.init_param)

    def get_module_result(self, module_name, points):
        module = self.module_list[module_name]
        placement_result = module.get_stable_placement(points)
        return placement_result

    def ani_callback(self, vis):
        if self.state == self.State.initalize:
            return False
        elif self.state == self.State.partial_view_sampling:
            if self.idx > self.total_num:
                self.state = self.State.inference
                self.idx = 0
                return False
            
            param = o3d.io.read_pinhole_camera_parameters(self.param_list[self.idx])
            
            partial_points = self.capture_tool.capture_mesh_to_points(self.mesh_file, 5000, param=param)
            camera = self.create_camera(param.intrinsic, param.extrinsic, scale=0.3)

            partial_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_points))
            partial_colors = np.zeros((partial_points.shape[0], 3))
            partial_colors[:, 0] = 1.0
            partial_pcd.colors = o3d.utility.Vector3dVector(partial_colors)

            self.vis.clear_geometries()
            self.vis.add_geometry(self.whole_pcd)
            self.vis.add_geometry(partial_pcd)
            self.vis.add_geometry(camera)
            self.reset_viewpoint()

            trg_idx = UOPSIM.downsampling(partial_points, 2048)
            partial_points = partial_points[trg_idx]

            self.partial_points.append(partial_points)
            self.frame_param.append(vis.get_view_control().convert_to_pinhole_camera_parameters())
            
            save_dir = "./partial_view_visualize"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.png".format(self.idx))

            save_dir = "./partial_points"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.pt".format(self.idx))
            torch.save(partial_points, save_path)

            self.idx += 1

            return False

        elif self.state == self.State.inference:
            if self.idx > self.total_num:
                self.state = self.State.end
                self.idx = 0
                return False

            partial_points = self.partial_points[self.idx]
            normalized_points, centroid, max_value = UOPSIM.normalize_point_cloud(partial_points)

            exp_result_dict = {}
            module_names = {
                "UOP": "uop",
                "RPF": "ransac",
                "CHSA": "trimesh",
                "BBF": "primitive",
            }
            exp_result_dict['input'] = {
                'points': normalized_points}

            for module_name in self.module_list.keys():
                exp_result = self.get_module_result(module_name, normalized_points)
                exp_result['input_points'] = partial_points
                exp_result_dict[module_names[module_name]] = exp_result
            
            # visualize
            save_dir = "./partial_result_visualize"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.png".format(self.idx))
            visualize_module_compare(exp_result_dict, save_path=save_path)
            
            self.idx += 1

            return False
            
        # vis.get_view_control().rotate(5.0, 0.0)
        return False        

if __name__=="__main__":
    data_root = "/home/ailab/Workspaces/_data/uop_data/ycb"
    obj_dir = "/home/ailab/Workspaces/_data/uop_data/ycb/002_master_chef_can"

    PointCloudVisualizer(obj_dir)

    # whole points + camera view



    # partial points + each module results
    




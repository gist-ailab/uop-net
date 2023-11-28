import os
import random
import glob
import natsort

import numpy as np
import open3d as o3d

from utils.file_utils import *
from utils.open3d_utils import sampling_points_from_mesh
from utils.capture_points_from_mesh import MeshCapture
from utils.matplotlib_visualize_utils import get_label2color

from dataset import UOPSIM


class PointCloudVisualizer:
    """Reference:
    http://www.open3d.org/docs/0.9.0/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer
    """
    class State:
        initialize = 0

        record_whole = 1
        record_label = 2
        
        init_partial = 3
        
        record_partial = 4

        record_end = 5

        set_pose = 90

    def __init__(self, uopsim_obj_dir):

        self.obj_dir = uopsim_obj_dir
        self.mesh_file = os.path.join(uopsim_obj_dir, "mesh_watertight.ply")
        self.cluster = load_pickle(os.path.join(uopsim_obj_dir, "inspected_zaxis.pkl"))

        self.prtial_points_list = [os.path.join(uopsim_obj_dir, 'partial', p) for p in os.listdir(os.path.join(uopsim_obj_dir, 'partial'))]
        
        self.whole_pcd = sampling_points_from_mesh(self.mesh_file, 30000)
        self.whole_points = np.asarray(self.whole_pcd.points, dtype=np.float32)

        # /home/ailab/Workspaces/_data/uop_data/ycb/002_master_chef_can/recorded_data/
        self.total_record_length = 450
        self.rotate_angle_x = 5.0   # : not a drgreee
        self.rotate_angle_y = 0.0
        self.zoom_z = 1.0              # +: set_zoom

        self.recorded_data = {      # : for save recorded data - whole, label, partial
            'whole': [],
            'label': [],
            'partial': [],
        }
        # self.recorded_data = {
        #     'whole': [],
        #     'label': [],
        #     'partial': [],
        #     'partial_1': [],
        #     'partial_2': [],
        # }

        self.save_root = os.path.join(self.obj_dir, 'recorded_data')
        for k in self.recorded_data.keys():
            os.makedirs(os.path.join(self.save_root, k), exist_ok=True)
        self.frame_idx = 0
        
        self.capture_tool = MeshCapture()

        # initialize visualizer
        self.state = self.State.initialize
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        
        # # add current PointCloud to window
        self.current_pcd = self.get_cuurent_pcd()
        self.vis.add_geometry(self.current_pcd)
        
        # load render option from file : RenderOption
        self.render_option = self.vis.get_render_option()
        
        # register callback function 
        self.vis.register_animation_callback(self.ani_callback)
        self.vis.register_key_callback(ord("S"), self.start_recording)
        self.vis.register_key_callback(ord("P"), self.capture_partial)

        self.vis.register_key_callback(ord("H"), self.capture_partial)  # : + up
        self.vis.register_key_callback(ord("B"), self.capture_partial)  # : + left
        self.vis.register_key_callback(ord("N"), self.capture_partial)  # : + down
        self.vis.register_key_callback(ord("M"), self.capture_partial)  # : + right
        self.vis.register_key_callback(ord("z"), self.zoom_in_view)  # : + zoom in
        self.vis.register_key_callback(ord("x"), self.zoom_out_view)  # : + zoom out

        # activate window
        self.vis.run()
    
    def get_cuurent_pcd(self):
        if self.state == self.State.initialize:
            self.whole_pcd.colors = o3d.utility.Vector3dVector(np.zeros((self.whole_points.shape[0], 3))+0.5)

            return self.whole_pcd
        
        elif self.state == self.State.record_whole:
            self.whole_pcd.colors = o3d.utility.Vector3dVector(np.zeros((self.whole_points.shape[0], 3))+0.5)

            return self.whole_pcd
        
        elif self.state == self.State.record_label:

            points = np.asarray(self.whole_pcd.points, dtype=np.float32)
            labels = UOPSIM.get_label_from_cluster(points, self.cluster)
            labels = UOPSIM._reorder_label(labels)
            label2color = get_label2color(labels)

            colors = np.zeros((points.shape[0], 4))
            unique_label = np.unique(labels)
            for ins_label in unique_label:
                colors[labels==ins_label] = label2color[int(ins_label)]

            self.whole_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            return self.whole_pcd

        elif self.state == self.State.record_partial:

            return self.current_partial
        

    def update_pcd(self):
        self.vis.clear_geometries()
        self.current_pcd = self.get_cuurent_pcd()
        self.vis.add_geometry(self.current_pcd)
        self.vis.reset_view_point(True)
        return False

    def start_recording(self, vis):
        ctr = vis.get_view_control()
        if self.state == self.State.initialize:
            self.init_param = ctr.convert_to_pinhole_camera_parameters()
            self.state = self.State.record_whole
        elif self.state == self.State.init_partial:
            self.state = self.State.record_partial
        return False

    def capture_partial(self, vis):
        points = self.capture_tool.capture_mesh_to_points(self.mesh_file, 5000)
        trg_idx = UOPSIM.downsampling(points, 10000)
        points = points[trg_idx]
        labels = UOPSIM.get_label_from_cluster(points, self.cluster)
        labels = UOPSIM._reorder_label(labels)

        label2color = get_label2color(labels)
        colors = np.zeros((points.shape[0], 4))
        unique_label = np.unique(labels)
        for ins_label in unique_label:
            colors[labels==ins_label] = label2color[int(ins_label)]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        self.current_partial = pcd
        self.vis.clear_geometries()
        self.vis.add_geometry(self.current_partial)
        self.init_camera_view()
        


    def init_camera_view(self):
        self.vis.reset_view_point(True)
        ctr = self.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(self.init_param)

    def rotate_view(self):
        self.vis.get_view_control().rotate(self.rotate_angle_x, self.rotate_angle_y)
    
    def zoom_in_view(self):
        print(f"zoom in: {self.zoom_z}")
        self.zoom_z += 0.1
        self.vis.get_view_control().set_zoom(self.zoom_z)
        self.vis.UpdateGeometry()
    def zoom_out_view(self):
        print(f"zoom out: {self.zoom_z}")
        self.zoom_z -= 0.1
        self.vis.get_view_control().set_zoom(self.zoom_z)
        self.vis.UpdateGeometry()

    def ani_callback(self, vis):
        if self.state == self.State.initialize:
            pass
        elif self.state == self.State.record_whole:
            save_dir = os.path.join(self.save_root, "whole")
            save_path = os.path.join(save_dir, "{}.png".format(self.frame_idx))
            self.vis.capture_screen_image(save_path)
            self.frame_idx += 1
            if self.frame_idx >= self.total_record_length:
                self.frame_idx = 0
                self.state = self.State.record_label
                self.update_pcd()
                self.init_camera_view()
            else:
                self.rotate_view()
        elif self.state == self.State.record_label:
            save_dir = os.path.join(self.save_root, "label")
            save_path = os.path.join(save_dir, "{}.png".format(self.frame_idx))
            self.vis.capture_screen_image(save_path)
            self.frame_idx += 1
            if self.frame_idx >= self.total_record_length:
                self.frame_idx = 0
                self.state = self.State.init_partial
            else:
                self.rotate_view()
        elif self.state == self.State.record_partial:
            save_dir = os.path.join(self.save_root, "partial")
            save_path = os.path.join(save_dir, "{}.png".format(self.frame_idx))
            self.vis.capture_screen_image(save_path)
            self.frame_idx += 1
            if self.frame_idx >= self.total_record_length:
                self.frame_idx = 0
                self.state = self.State.record_end
                self.init_camera_view()
            else:
                self.rotate_view()
        
        # + end
        elif self.state == self.State.record_end:
            print("-->> close window")
            self.vis.destroy_window()
            
        return False


# 녹화 State
# 초기 자세를 정한다
# 녹화 시작
# 1. 물체를 회전시킨다
# 2. 물체 바꾼다




# if __name__=="__main__":
#     data_root = "/home/ailab/Workspaces/_data/uop_data/ycb"

#     obj_dir = "/home/ailab/Workspaces/_data/uop_data/ycb/002_master_chef_can"
#     PointCloudVisualizer(obj_dir)



if __name__=="__main__":
    data_root = "/home/ailab-ur5/_Workspaces/_data/uop_data/"
    dataset ="ycb"
    dataset_root = os.path.join(data_root, dataset)

    obj_list = glob.glob(dataset_root+"/*")
    sorted_obj_list = natsort.natsorted(obj_list)

    for idx, one_obj in enumerate(sorted_obj_list):
        if idx < 47:
            continue
        # obj_dir = f"{data_root}/{dataset}/{one_obj}"
        PointCloudVisualizer(one_obj)
# 포인트 개수가 부족할때 다운샘플시, 터지는 문제가있음




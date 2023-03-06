import os
import random
import argparse

import numpy as np
import open3d as o3d
from pathlib import Path
import pyfastnoisesimd as fns
import cv2

# = From real env -> rgb intrinsic, depth2rgb(in ros ppoint cloud capture)
# AZURE_INTRINSIC = [[973.5092163085938, 0.0, 1019.8231811523438], [0.0, 973.4360961914062, 779.4862670898438], [0.0, 0.0, 1.0]]
AZURE_INTRINSIC = [
    [973.5092163085938, 0.0, 1023.5],
    [0.0, 973.4360961914062, 767.5], 
    [0.0, 0.0, 1.0]]
AZURE_WIDTH = 2048
AZURE_HEIGHT = 1536     # : 4:3, 1536p NWOF


class MeshCapture:
    def __init__(self, param_dir=os.path.join(str(Path(__file__).parent.absolute()), "o3d_cam_param")):
        self.vis = o3d.visualization.Visualizer()
        self.param_list = [os.path.join(param_dir, p) for p in os.listdir(param_dir)]
        self.vis.create_window(width=AZURE_WIDTH, 
                               height=AZURE_HEIGHT)

    def get_random_param(self):
        param_path =  random.choice(self.param_list)
        return o3d.io.read_pinhole_camera_parameters(param_path)
    
    def capture_mesh_to_points(self, mesh_file, min_num):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        self.vis.add_geometry(mesh, reset_bounding_box=True)
        self.update_view()
        
        # initialize window
        is_captured = False
        down_sample = True
        while not is_captured:
            self.param = self.get_random_param()
            view_ctr = self.vis.get_view_control()
            view_ctr.convert_from_pinhole_camera_parameters(self.param)
            self.update_view()
            
            # self.vis.capture_screen_image("test.png")
            
            # capture
            points = self.capture_point_cloud(down_sample)
            down_sample = False
            if points.shape[0] > min_num:
                is_captured = True
                break

        self.vis.clear_geometries()
        self.update_view()
        
        return points

    def update_view(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_point_cloud(self, down_sample=True):
        depth = self.vis.capture_depth_float_buffer(do_render=True)
        depth = np.asarray(depth)
        
        obj_mask = depth > 0
        # depth noise
        depth = self.PerlinDistortion(depth, AZURE_WIDTH, AZURE_HEIGHT)
        depth[~obj_mask] = 0
        # # depth mask
        # obj_mask = np.uint8(obj_mask*255)
        # contour, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contr = contour[0]
        
        # target_idx = random.choice(contr)[0]
        # sx=max(target_idx[0]-20, 0)
        # ex=min(target_idx[0]+20, AZURE_HEIGHT)
        # sy=max(target_idx[1]-20, 0)
        # ey=min(target_idx[1]+20, AZURE_WIDTH)
        # depth[sx:ex, sy:ey] = 0
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), 
                                                              self.param.intrinsic, 
                                                              self.param.extrinsic)
        if down_sample:
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            
        pcd , _ = pcd.remove_statistical_outlier(10, 0.01)
        # pcd.rotate(rot_mat)
        points = np.asarray(pcd.points)
        
        return points

    def stop(self):
        self.vis.destroy_window()
        
    @staticmethod    
    def PerlinDistortion(image, width, height):
        def perlin_noise(frequency, width, height):
            noise = fns.Noise()
            noise.NoiseType = 2 # perlin noise
            noise.frequency = frequency
            result = noise.genAsGrid(shape=[height, width], start=[0,0])
            return result
        """
        """
        # sample distortion parameters from noise vector
        fx = np.random.uniform(0.0001, 0.1)
        fy = np.random.uniform(0.0001, 0.1)
        fz = np.random.uniform(0.01, 0.1)
        wxy = np.random.uniform(0, 10)
        wz = np.random.uniform(0, 0.005)
        cnd_x = wxy * perlin_noise(fx, width, height)
        cnd_y = wxy * perlin_noise(fy, width, height)
        cnd_z = wz * perlin_noise(fz, width, height)

        cnd_h = np.array(list(range(height)))
        cnd_h = np.expand_dims(cnd_h, -1)
        cnd_h = np.repeat(cnd_h, width, -1)
        cnd_w = np.array(list(range(width)))
        cnd_w = np.expand_dims(cnd_w, 0)
        cnd_w = np.repeat(cnd_w, height, 0)

        noise_cnd_h = np.int16(cnd_h + cnd_x)
        noise_cnd_h = np.clip(noise_cnd_h, 0, (height - 1))
        noise_cnd_w = np.int16(cnd_w + cnd_y)
        noise_cnd_w = np.clip(noise_cnd_w, 0, (width - 1))

        new_img = image[(noise_cnd_h, noise_cnd_w)]
        new_img = new_img = new_img + cnd_z
        return new_img.astype(np.float32)

def check_o3d_headless():
    if not o3d._build_config['ENABLE_HEADLESS_RENDERING']:
        print("Headless rendering is not enabled. "
              "Please rebuild Open3D with ENABLE_HEADLESS_RENDERING=ON")
        exit(1)

        
def generate_cam_parameters(sample_mesh, save_dir, num=1000):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=AZURE_WIDTH, height=AZURE_HEIGHT, left=50, top=50)
    view_ctr = vis.get_view_control()
    
    vis.poll_events()
    vis.update_renderer()
    
    mesh = o3d.io.read_triangle_mesh(sample_mesh)
    vis.add_geometry(mesh, reset_bounding_box=True)
    
    vis.poll_events()
    vis.update_renderer()
    
    param = view_ctr.convert_to_pinhole_camera_parameters()
    param.intrinsic.intrinsic_matrix = AZURE_INTRINSIC
    view_ctr.convert_from_pinhole_camera_parameters(param)
    
    vis.poll_events()
    vis.update_renderer()
    
    def save_cam_param(idx):
        print(idx)
        save_path = os.path.join(save_dir, "{}.json".format(idx))
        param = view_ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(save_path, param)
    
    # about 4000 -> 360 degree
    def random_x_rotation():
        # rot_x = np.random.rand(1)*10
        rot_x = np.random.rand(1)*400
        view_ctr.rotate(rot_x, 0)
    
    def random_y_rotation():
        rot_y = np.random.rand(1)*400
        view_ctr.rotate(0, rot_y)

    def random_scale():
        scale = 1 + (np.random.rand(1) - 0.5)/100
        # scale = 1.1
        view_ctr.scale(scale)
    
    for idx in range(1, num+1):
        random_x_rotation()
        # time.sleep(0.5)
        if idx % 10 == 0:
            random_y_rotation()
        
        if idx % 100 == 0:
            random_scale()
        
        vis.poll_events()
        vis.update_renderer()
        save_cam_param(idx)

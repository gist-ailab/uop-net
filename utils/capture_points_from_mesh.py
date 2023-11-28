
import os
import random
import numpy as np
import open3d as o3d
from pathlib import Path

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
        self.param_dir = param_dir
        self.param_list = [os.path.join(param_dir, p) for p in os.listdir(param_dir)]
        
    def get_random_param(self):
        param_path =  random.choice(self.param_list)
        return o3d.io.read_pinhole_camera_parameters(param_path)
    
    def capture_mesh_to_points(self, mesh_file, min_num, param=None):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        # initialize window
        if param is None:
            param = self.get_random_param()
        else:
            pass
        
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=o3d.cuda.pybind.core.Tensor(param.intrinsic.intrinsic_matrix), # (3, 3)
            extrinsic_matrix=o3d.cuda.pybind.core.Tensor(param.extrinsic), # (4, 4) array
            width_px= param.intrinsic.width,
            height_px=param.intrinsic.height
        )
        # We can directly pass the rays tensor to the cast_rays function.
        ans = scene.cast_rays(rays)
        hit = ans['t_hit'].isfinite()
        points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
        pcd = o3d.t.geometry.PointCloud(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        points = np.asarray(pcd.to_legacy().points)
        return points
        
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


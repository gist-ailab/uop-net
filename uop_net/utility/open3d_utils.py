import sys
import numpy as np
import open3d as o3d

from os import path, read
from os.path import join
from typing import Optional

def read_point_cloud(pc_file) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pc_file)
    return pcd

def read_mesh(mesh_file) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    return mesh

def save_point_cloud(pcd: o3d.geometry.PointCloud, save_path: str) -> None:
    o3d.io.write_point_cloud(save_path, pcd)

def save_mesh(mesh: o3d.geometry.TriangleMesh, save_path: str) -> None:
    o3d.io.write_triangle_mesh(save_path, mesh)

def convert_numpy_to_point_cloud(points : np.ndarray, normal: Optional[np.ndarray] = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.normalize_normals()
    return pcd

def set_pcd_color(pcd : o3d.geometry.PointCloud, colors : np.ndarray):
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_numpy_to_ply(points, save_path, normal=None):
    """[summary]
    Args:
        points (Np, 3): [points]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.normalize_normals()
    o3d.io.write_point_cloud(save_path, pcd)

def generate_mesh_from_points(points, normal, method="BPA"):
    """[summary]
    
    Args:
        points: np.ndarray(number of points, 3)
        methods: ["BPA" or "poisson"]
    
    Reference:
    
    https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    if method == "BPA":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
        
        dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
        
        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()
        mesh = dec_mesh
    elif method == "poisson":
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        mesh = poisson_mesh
    
    return mesh

def sampling_points_from_mesh(mesh_file, number_of_points=2048, method="uniform"):
    """
    Reference: 
    http://www.open3d.org/docs/release/tutorial/geometry/mesh.html
    """
    mesh = read_mesh(mesh_file)
    if method=="uniform":
        pcd = mesh.sample_points_uniformly(number_of_points)
    elif method=="poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=5)
    elif method=="uniform-poisson":
        pcd = mesh.sample_points_uniformly(number_of_points)
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, pcl=pcd)
    
    return pcd

def plane_fitting_from_points(points : np.ndarray):
    """get plane from points
    Args:
        points (np.ndarray): [(n, 3) target points]
    Returns:
        plane_model: a, b, c, d of equation {ax+by+cz=d}
        inliers: index of inlier points
    """
    assert len(points) > 3, "There is at most 3 points to fitting plane but got {}".format(len(points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.001,
                                            ransac_n=3,
                                            num_iterations=1000)
    return plane_model, inliers
  
def hidden_point_removal(pcd, camera_location, radius):
    re = pcd.hidden_point_removal(camera_location, radius)

class PointCloudVisualizer:
    """Reference:
    
    http://www.open3d.org/docs/0.9.0/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer
    
    """

    def get_custom_pcd(self) -> o3d.geometry.PointCloud:
        """TODO: if there is preprocess for cloud file
        
        # using self.object_idx for get current object
        point_cloud_file = self.object_file_list[self.object_idx]
        
        # example for npy files
        pcd = o3d.geometry.PointCloud()
        xyz = np.load(point_cloud_file)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        return pcd
        """
        raise NotImplementedError
  
    def get_custom_pcd_color(self) -> o3d.utility.Vector3dVector:
        """TODO: if there is additional color
        current_pcd_color = self.color_list[self.object_idx]
        return o3d.utility.Vector3dVector(current_pcd_color)
        
        
        Raises:
            NotImplementedError: [description]
        Returns:
            [o3d.utility.Vector3dVector()]: [description]
        """
        
        raise NotImplementedError
    
    def __init__(self, point_cloud_file_list, custom_pcd=False, custom_color=False):
        self.object_file_list = point_cloud_file_list
        self.custom_pcd = custom_pcd
        self.custom_color = custom_color
        
        self.object_idx = -1
        self.object_num = len(self.object_file_list)
        self.current_pcd = self.get_next_pcd()
        
        # initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(left=600) # margin
        
        # # add current PointCloud to window
        self.vis.add_geometry(self.current_pcd)
        
        # load render option from file : RenderOption
        self.render_option = self.vis.get_render_option()
        self.render_option.load_from_json(join(path.dirname( path.abspath(__file__) ), "point_cloud_render_option.json"))
        
        # register callback function 
        # self.vis.register_animation_callback(self.rotate_view)
        self.vis.register_key_callback(ord("N"), self.update_pcd)
        self.vis.register_key_callback(ord("B"), self.previous_pcd)
        # activate window
        self.vis.run()
    
    def get_next_pcd(self):
        self.object_idx += 1
        if not self.object_idx < self.object_num:
            print("No more Objects ==> Exit Program!!")
            exit()
        
        # next object file from list    
        point_cloud_file = self.object_file_list[self.object_idx]
        print("Get Next Object: {}".format(point_cloud_file))
        # create PointCloud
        if self.custom_pcd:
            pcd = self.get_custom_pcd()
        else:
            supported_format = [".xyz", ".ply", ".pcd", ".pts", ".xzyrgb", ".xyzn"] # http://www.open3d.org/docs/0.6.0/tutorial/Basic/file_io.html?highlight=read_point_cloud#point-cloud
        
        if path.splitext(point_cloud_file)[-1] in supported_format:
            pcd = o3d.io.read_point_cloud(point_cloud_file)
        
        else:
            assert False, "Not Supported Type {}".format(path.split(point_cloud_file)[-1])
        
        # if there is custom color
        if self.custom_color:
            pcd.colors = self.get_custom_pcd_color()

        return pcd
        
    def update_pcd(self, vis):
        vis.clear_geometries()
        self.current_pcd = self.get_next_pcd()
        vis.add_geometry(self.current_pcd)
        vis.reset_view_point(True)
        
        return False
    
    def previous_pcd(self, vis):
        vis.clear_geometries()
        self.object_idx -= 2
        self.current_pcd = self.get_next_pcd()
        vis.add_geometry(self.current_pcd)
        vis.reset_view_point(True)
        
        return False
    
    def rotate_view(self, vis):
        ctr = vis.get_view_control()
        ctr.rotate(5.0, 0.0)
        return False

    
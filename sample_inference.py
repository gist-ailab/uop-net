import argparse
import os
import torch

from placement_module import load_placement_module

from utils.capture_points_from_mesh import MeshCapture
from utils.open3d_utils import down_sample_points
from utils.matplotlib_visualize_utils import visualize_exp_result

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='003_cracker_box')
    args = parser.parse_args()
    
    
    # Target Object
    sample_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample')
    sample_obj_dir = os.path.join(sample_root, args.object)
    result_dir = os.path.join(sample_obj_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
        
    # Placement Module
    module_list = {
        "UOP": "uop",
        "RPF": "ransac",
        "CHSA": "trimesh",
        "BBF": "primitive",
    }
    
    # Capture Tool for partial point cloud
    capture_tool = MeshCapture()
    mesh_file = os.path.join(sample_obj_dir, "mesh.ply")
    points = capture_tool.capture_mesh_to_points(mesh_file, min_num=2048)
    target_idx = down_sample_points(points, 2048)
    partial_points = points[target_idx]

    for method, module_name in module_list.items():
        placement_module = load_placement_module(module_name)
        
        inference_result = placement_module.get_stable_placement(partial_points)
        result_path = os.path.join(result_dir, "{}.pt".format(method))
        torch.save(inference_result, result_path)
        visualize_path = os.path.join(result_dir, "{}.png".format(method))
        visualize_exp_result(exp_result=inference_result,
                             module=module_name,
                             save_path=visualize_path)
        
    
    
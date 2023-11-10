
import os
import argparse

import trimesh
import numpy as np
import subprocess

from plotly import graph_objects as go


from pyrep import PyRep

from utils.pyrep_utils import convert_mesh_to_scene_object
from utils.file_utils import load_yaml_to_dic, load_pickle
from utils.plotly_visualize_utils import plot_mesh, plot_points_with_label

from uop_sim.data_generator import UOPDataGenerator
from uop_sim.labeling import get_instance_label_from_clustered_info


def normalize_point_cloud(points : np.ndarray) -> np.ndarray:
    assert points.shape[0] > 0
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_value = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_value
    
    return points, centroid, max_value

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    points, centroid, max_value = normalize_point_cloud(mesh.vertices)
    mesh.vertices = points
    return mesh

def convert_to_watertight(manifold_path, input_mesh, output_mesh):
    print(f'watertight running on "{input_mesh}"')
    #!TODO: This custom path should be modified after sudo make install
    cmd = f'{manifold_path}/manifold {input_mesh} {output_mesh}'
    retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    if retobj.returncode != 0:
        print(f'manifold failed on "f{input_mesh}"')
        if retobj.stdout != '': print(f'{retobj.stdout}')
        if retobj.stderr != '': print(f'{retobj.stderr}')
    mesh = trimesh.load_mesh(output_mesh)
    os.system(f"rm {input_mesh}")
    os.system(f"rm {os.path.dirname(input_mesh)}/material*")
    os.system(f"rm {output_mesh}")
    
    return mesh


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default='003_cracker_box')
    args = parser.parse_args()
    
    # Prepare Path 
    ycb_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models', 'ycb')
    manifold_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'thirdparty/Manifold/build')
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uop_sim/config.yaml')
    
    
    # Save Path
    sample_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample')

    # Target Object
    trg_obj_dir = os.path.join(ycb_root, args.object)
    sample_obj_dir = os.path.join(sample_root, args.object)
    os.makedirs(sample_obj_dir, exist_ok=True)
    
    #region Preprocess YCB Object
    
    #1. load mesh file
    mesh_file = os.path.join(trg_obj_dir, "google_16k", "textured.obj")
    mesh = trimesh.load(mesh_file)
    
    #2. check if the mesh is watertight
    if mesh.is_watertight:
        print("The mesh is already watertight.")
    else:
        input_mesh = os.path.join(sample_obj_dir, "mesh_input.obj")
        output_mesh = os.path.join(sample_obj_dir, "mesh_out.obj")
        mesh.export(input_mesh)
        mesh = convert_to_watertight(manifold_path, input_mesh, output_mesh)
    
    mesh = normalize_mesh(mesh)
    mesh.export(os.path.join(sample_obj_dir, "mesh.ply"))
    fig = go.Figure(data=[plot_mesh(mesh)])
    fig.write_html(os.path.join(sample_obj_dir, 'visualize_mesh.html'))
    
    #3. generate coppeliasim scene model
    pr = PyRep()
    pr.launch(headless=True)
    pr.start()
    
    mesh_file = os.path.join(sample_obj_dir, "mesh.ply")
    save_path = os.path.join(sample_obj_dir, "model.ttm")
    obj_name = args.object
    convert_mesh_to_scene_object(pr, mesh_file, save_path,
                                 bbox_size=0.2, target_name=obj_name)
    
    pr.stop()
    pr.shutdown()

    #endregion

    #region Generate Placement Data
    cfg = load_yaml_to_dic(config_path)
    cfg['data_type'] = "ycb sample"
    cfg['headless'] = True
    data_generator = UOPDataGenerator(cfg)
    
    # #1. samplilng stable pose
    model_path = os.path.join(sample_obj_dir, "model.ttm")
    save_path = os.path.join(sample_obj_dir, "stable_pose.pkl")
    data_generator._simulate_stability(model_path, save_path)
    
    # #2. clustering stable pose
    stability_file = os.path.join(sample_obj_dir, "stable_pose.pkl")
    save_path = os.path.join(sample_obj_dir, "placement_axis.pkl")
    data_generator._clustering_stability(stability_file, save_path)
    
    #3. inspect clustering result
    data_generator.convert_to_labeling_env()
    
    model_path = os.path.join(sample_obj_dir, "model.ttm")
    cluster_info = load_pickle(os.path.join(sample_obj_dir, "placement_axis.pkl"))
    save_path = os.path.join(sample_obj_dir, "label.pkl")
    data_generator._inspecting_cluster(model_path, cluster_info, save_path)
    
    data_generator.stop()
    
    #endregion

    #region Visualize Placement Data
    mesh_file = os.path.join(sample_obj_dir, "mesh.ply")
    cluster_info = load_pickle(os.path.join(sample_obj_dir, "label.pkl"))
    mesh = trimesh.load(mesh_file)
    labels = get_instance_label_from_clustered_info(mesh_file, cluster_info)
    _, labels = np.unique(labels, return_inverse=True)
    
    vis_data = []
    vis_data += plot_points_with_label(mesh.vertices, labels)
    
    
    fig = go.Figure(data=vis_data)
    fig.write_html(os.path.join(sample_obj_dir, 'visualize_label.html'))
    print()
    
    
    #endregion

    
    
    
    
    

    
    
    

import open3d as o3d
import numpy as np

from utils.file_utils import *
from utils.open3d_utils import *


def min_max_normalize(ndarray):
    nmax, nmin = ndarray.max(), ndarray.min()
    normalized = (ndarray - nmin) / (nmax - nmin)
    
    return normalized

def clustering_sampled_pose(stability_file):
    stability = load_pickle(stability_file)

    orientation_list = []
    stable_score_list = []
    stable_matrix_list = []
    start_z_list = []
    last_z_list = []
    
    for orientation, (stable_score, stable_matrix, start_z, last_z) in stability.items():
        if stable_score > 100:
            continue
        orientation_list.append(orientation)
        stable_score_list.append(stable_score)
        stable_matrix_list.append(stable_matrix)
        start_z_list.append(start_z)
        last_z_list.append(last_z)
        
    ori = np.array(orientation_list)
    last_z = np.array(last_z_list)
    
    # clustering
    z_axis_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(last_z))
    labels = np.array(z_axis_pcd.cluster_dbscan(eps=0.1, min_points=5, print_progress=False))

    # filter noise: 0, 1, 2, ... 
    filtered_ori = ori[labels >= 0]
    filtered_z = last_z[labels >= 0]
    filtered_labels = labels[labels >= 0]
    
    if len(filtered_z) == 0:
        return {}
    max_label = filtered_labels.max()
    unique_labels = np.unique(filtered_labels)
    
    # combine delta < 25 z_axis
    
    target_z_list = []
    
    for label in unique_labels:
        z_axis_clusters = filtered_z[filtered_labels==label]

        # mean of cluster
        mean_z_axis = np.mean(z_axis_clusters, axis=0)
        is_overlap = False
        for idx, target_z in enumerate(target_z_list):
            dot_p = np.dot(target_z, mean_z_axis)/(np.linalg.norm(target_z)*np.linalg.norm(mean_z_axis))
            if dot_p < 0:
                continue
            else:
                delta = np.arccos(dot_p)*(180/np.pi)
                if delta < 25:
                    is_overlap = True
                    new_z = target_z + mean_z_axis
                    target_z_list[idx] = new_z
        if not is_overlap:
            target_z_list.append(mean_z_axis)

    clustered_info = {}
        
    for label, target_z_axis in enumerate(target_z_list):
        target_idx = np.argmin(np.linalg.norm(filtered_z - target_z_axis, axis=1))
        target_ori = filtered_ori[target_idx]

        clustered_info[tuple(target_ori)] = {
            "label": label,
            "matrix": stability[tuple(target_ori)][1],
            "z_axis": target_z_axis,
        }

    return clustered_info
        
def get_instance_label_from_clustered_info(pc_file, clustered_info):
    # hyper parameter
    labeling_tolerance = 0.05 # labeled points from 0 ~ labeling_tolerance
    
    pcd = o3d.io.read_point_cloud(pc_file)
    
    points_cloud = np.asarray(pcd.points)
    points_cloud = points_cloud - np.mean(points_cloud, axis=0)
    
    points_cloud_labels = np.zeros((points_cloud.shape[0]))
    
    for info in clustered_info.values():
        mean_z_axis = info['z_axis']
        label = int(info['label'])
        dot_product = np.dot(points_cloud, mean_z_axis)
        dot_product = min_max_normalize(dot_product)

        # label: 1, 2, 3 ...
        points_cloud_labels[dot_product<labeling_tolerance] = label + 1

    return points_cloud_labels

def get_instance_mask_by_zaxis(points_cloud, z_axis, labeling_tolerance=0.05):
    dot_product = np.dot(points_cloud, z_axis)
    dot_product = min_max_normalize(dot_product)

    return dot_product < labeling_tolerance


import os
import re
import copy
import random   # TODO : is used?

import torch.utils.data as data
import torch

import numpy as np
from tqdm import tqdm
import point_cloud_utils as pcu

from utils.file_utils import *
from utils.open3d_utils import *
from utils.capture_points_from_mesh import MeshCapture


class UOPSIM(data.Dataset):
    def __init__(self, root,
                   num_points=2048, under=-1, sampling='random', val=None,
                   partial=True, label_cut=10, seed=None, non_zero=True,
                   projection=False, max_instances=32):
        """ uop dataset initialize 
        Args:
            root: uop dataset root(data generator save root)
            num_points(default: 2048): num of points
            partial(default: False): True => partial sample points
            partial_ratio(default: 2000): larger more partial points
            projection(default: False): True => add projection value to input points(N, 4)
            max_instances(default: 100): Affect instance mask size [N, 100]
        """

        self.data_root = root
        
        self.num_points = num_points
        self.sampling = sampling

        self.partial = partial
        
        self.label_cut = label_cut

        self.projection = projection

        self.max_instances = max_instances

        self.seed = seed
        self.under = under

        self.capture_tool = MeshCapture()

        self.data_list = get_dir_list(self.data_root)
        self.points_cloud = []
        self.labels_cloud = []
        self.cluster_info = []
        self.mesh_cloud = [] 
        self.object_names = []
        
        total_names = []
        self.total_name_list = []
        
        for data_dir in tqdm(self.data_list, 'data loading'):
            if not self.check_file_existancy(data_dir):
                continue
      
            obj_name = get_dir_name(data_dir)
            if val is not None:
                if not obj_name in val:
                    continue
            alpahbet_len = len(re.findall('[a-zA-Z_]', get_dir_name(data_dir)))
            total_names.append(get_dir_name(data_dir)[:alpahbet_len])
      
            if under > 0:
                if not self.check_under(obj_name, under):
                    continue
            
            label = self._load_label(os.path.join(data_dir, 'label_inspected.pkl'))
            cluster = self._load_cluster(os.path.join(data_dir, 'cluster_inspected.pkl'))
            
            if non_zero:
                if np.max(label) == 0:
                    continue
            
            self.object_names.append(obj_name)
            self.points_cloud.append(self._load_point_cloud(os.path.join(data_dir, 'point_cloud.ply')))
            self.labels_cloud.append(label)
            self.cluster_info.append(cluster)
            self.mesh_cloud.append(os.path.join(data_dir, 'mesh_watertight.ply'))
        
        self.total_name_list = np.unique(total_names)
        self.max_instances = int(max_instances)
        self.dataset_size = len(self.object_names)
        print("Whole Dataset Size: {}".format(self.dataset_size))
    
    @staticmethod
    def get_object_cat(object_name):
        idx = -1
        while object_name[idx].isnumeric():
            idx -= 1
        if idx == -1:
            return object_name
        else:
            return object_name[:idx]
    
    @staticmethod
    def check_under(object_name, under):
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
    
    def __len__(self):
        return self.dataset_size

    def check_file_existancy(self, object_dir):
        file_list = os.listdir(object_dir)
        if not 'point_cloud.ply' in file_list:
            return False
        if not 'cluster_inspected.pkl' in file_list:
            return False
        if not 'label_inspected.pkl' in file_list:
            return False
        if not 'mesh_watertight.ply' in file_list:
            return False
        return True

    @staticmethod
    def _load_point_cloud(file_path):
        return np.asarray(read_point_cloud(file_path).points, dtype=np.float32)

    @staticmethod
    def _load_label(file_path):
        return load_pickle(file_path)

    @staticmethod
    def _load_cluster(file_path):
        cluster = load_pickle(file_path)
        temp = {}
        for v in cluster.values():
            temp[v['label']+1] = v
        return temp
    
    @staticmethod
    def _reorder_label(label):
        unique_labels, inv = np.unique(label, return_inverse=True)
        if unique_labels[0] == 0:
            return inv
        else:
            temp = np.zeros_like(label)
            label_idx = 1
            for unique_label in unique_labels:
                temp[label==unique_label] = label_idx
                label_idx += 1
            return temp
    
    @staticmethod
    def _min_max_normalize(ndarray):
        nmax, nmin = ndarray.max(), ndarray.min()
        normalized = (ndarray - nmin) / (nmax - nmin)
    
        return normalized

    @staticmethod
    def normalize_point_cloud(points : np.ndarray) -> np.ndarray:
        assert points.shape[0] > 0
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_value = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / max_value
        
        return points, centroid, max_value

    @staticmethod
    def unnormalize_point_cloud(points : np.ndarray, centroid, max_value) -> np.ndarray:
        assert points.shape[0] > 0
        return (points * max_value) + centroid
    
    @staticmethod
    def farthest_point_sample(points, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        xyz = torch.from_numpy(points).unsqueeze(0)
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids[0]
    
    @staticmethod
    def downsampling(points: np.ndarray, num_sample, method='random') -> np.ndarray:
        assert len(points) >= num_sample
    
        if method == 'poisson':
            target_idx = pcu.downsample_point_cloud_poisson_disk(points, num_samples=num_sample)[:num_sample]
            if len(target_idx) < num_sample:
                target_idx = np.r_[target_idx, np.setdiff1d(np.arange(len(points)), target_idx)[:num_sample-len(target_idx)]]
        
        elif method == 'random':
            target_idx = np.random.choice(range(len(points)), num_sample)
        
        elif method == 'fps':
            target_idx = UOPSIM.farthest_point_sample(points, num_sample)
        
        else:
            raise NotImplementedError
        
        return target_idx
    
    @classmethod
    def get_label_from_cluster(cls, points: np.ndarray, cluster_info):
        instance_num = len(cluster_info)
        labels = np.ones((points.shape[0], instance_num+1))
        
        instance_labels = []
        
        for label_idx, info in enumerate(cluster_info.values()):
            mean_z_axis = info['z_axis']
            label = int(info['label'])
            instance_labels.append(label)
            dot_product = np.dot(points, mean_z_axis)
            dot_product = cls._min_max_normalize(dot_product)
            
            dot_product[dot_product>0.075] = 1
            labels[:,label_idx+1] = dot_product

        labels = np.argmin(labels, axis=1)
        ins_labels = np.zeros_like(labels)
        for label_idx, label in enumerate(instance_labels):
            ins_labels[labels==label_idx+1] = label + 1
    
        return ins_labels
    
    @staticmethod
    def filtering_label(points, labels, cluster_info):
        filtered = copy.deepcopy(labels)
        instance_labels = np.unique(labels)
        if instance_labels[0] == 0:
            instance_labels = instance_labels[1:]
        for ins_label in instance_labels:
            raw_normal = cluster_info[int(ins_label)]['z_axis']
            targets = points[labels==ins_label]
            
            if targets.shape[0] > 3:
                plane, _ = plane_fitting_from_points(targets)
                normal = plane[:3]
                raw_normal = raw_normal / np.linalg.norm(raw_normal)
                normal = normal / np.linalg.norm(normal)
                dot_ = np.dot(raw_normal, normal)
                angle = (np.arccos(abs(dot_))/np.pi)*180
                if angle > 10:
                    print(angle)
                    filtered[labels==ins_label] = 0
            else:
                filtered[labels==ins_label] = 0
        
        return filtered
    
    def _partial_sampling_o3d(self, points, labels):
        pcd = convert_numpy_to_point_cloud(points)
        
        center = np.mean(points, axis=0)
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        cam_distance = np.random.uniform(1, 3)
        cam_radius = np.random.uniform(10, 100)
        camera_direction = np.random.rand(3) - 0.5
        camera_direction /= np.linalg.norm(camera_direction)
        camera = center + diameter* camera_direction * cam_distance
        _, pt_map = pcd.hidden_point_removal(camera, diameter * cam_radius)
        del pcd
        if len(pt_map) < self.num_points:
            pt_map = np.r_[pt_map, np.setdiff1d(np.arange(points.shape[0]), pt_map)[:self.num_points - len(pt_map)]]
        
        points = points[pt_map]
        labels = labels[pt_map]
        
        if self.projection:
            proj = self._min_max_normalize(np.dot(points, camera_direction))
            proj = np.expand_dims(proj, axis=-1).astype(np.float32)
            points = np.concatenate([points, proj], axis=1)
        
        return points, labels

    def _partial_sampling_mesh(self, whole_points, mesh_file, cluster_info):
        captured_points = self.capture_tool.capture_mesh_to_points(mesh_file, min_num=2048)
        labels = np.zeros((captured_points.shape[0]))
        
        merged_points = np.concatenate([whole_points, captured_points], axis=0)
        merged_labels = self.get_label_from_cluster(merged_points, cluster_info)
        labels = merged_labels[-captured_points.shape[0]:]
        
        return captured_points, labels
    
    def __getitem__(self, i):
        """_summary_

        Args:
                i (int): index of data

        Returns:
            points(ndarray, (2048, 3)): points xyz,
            sem_labels(ndarray, (2048,)): semantic labels of points(0: unstable, 1: stable),
            ins_labels(ndarray, (2048, 100)): one hot instance masks(0: unstable instance, others: stable instance),
            size(int): total instance
            scene_model(str): scene model file path
            object_name(str): object name
            centroid(ndarray, (3)): centroid of raw points
            max_value(float): max_value of raw points
                
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        points = self.points_cloud[i]
        labels = self.labels_cloud[i]
        cluster = self.cluster_info[i]
        mesh_file = self.mesh_cloud[i]

        # partial sampling
        if self.partial:
            # points, labels = self._partial_sampling(points, labels)
            points, labels = self._partial_sampling_mesh(points, mesh_file, cluster)
            
        # down sampling 
        target_idx = self.downsampling(points, num_sample=self.num_points, method=self.sampling)
        points = points[target_idx].astype(np.float32)
        labels = labels[target_idx]
        
        # Remove too much cut offed instance labels
        # labels = self.filtering_label(points, labels, cluster)
        # reorder
        labels = self._reorder_label(labels)
        
        # normalize
        points[:, :3], centroid, max_value = self.normalize_point_cloud(points[:, :3])
        
        # instance mask (one hot of instance label) (0, 1, 2 ..) 0 is unstable
        masks = np.zeros((points.shape[0], self.max_instances), dtype=np.float32) # instance mask for each point 
        masks[np.arange(points.shape[0]), labels.astype(int)] = 1 # instance one hot
        
        # semantic label (0 or 1)
        sem_labels = np.where(labels > 0, 1.0, 0.0).astype(np.float32)
        
        return {
            'points': points,
            'sem_labels': sem_labels,
            'ins_labels': masks,
            'size': np.unique(labels).size, # instance num of target points
            "object_name": self.object_names[i],
            "centroid": centroid,
            "max_value": max_value
        }


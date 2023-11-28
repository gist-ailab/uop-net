import numpy as np
from sklearn import cluster
import torch


from uop_net import load_model
from dataset import UOPSIM
from utils.placement_utils import *
from utils.meanshift.mean_shift_gpu  import MeanShiftEuc
from utils.trimesh_utils import get_stable_transform


class PlacementModule:
    def __init__(self):
        pass
    
    def get_stable_placement(self, points):
        raise NotImplementedError
    
    @staticmethod
    def get_best_plane_info(points, ins_preds):
        """ get best plane from normalized points and mask

        Args:
            points (_type_): _description_
            ins_preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        best_plane_info = {}
        best_inlier = 0
        
        instance_label = np.unique(ins_preds)
        for instance_idx in instance_label:
            if not instance_idx > 0:
                continue
            target_mask = ins_preds == instance_idx
            target_points = points[target_mask]
            if target_points.shape[0] < 50:
                continue
    
            try:
                plane_model, inliers = plane_fitting_from_points(target_points)
            except:
                continue
        
            if best_inlier > len(inliers):
                continue
            
            best_inlier = len(inliers)
            plane_center = np.mean(target_points[inliers], axis=0)
            matrix = calculate_placement_transform(plane_model, plane_center, np.zeros(3))
            best_plane_info ={
                "instance_idx": instance_idx,
                "inlier_ratio": len(inliers)/len(target_points),
                "inlier": inliers,
                "plane_eq": plane_model,
                "plane_center": plane_center,
                "rotation": matrix,
            }
        
        return best_plane_info


class GTOracle(PlacementModule):

    def get_stable_placement(self, points, ins_preds):
        
        plane_info = self.get_best_plane_info(points, ins_preds)
        
        exp_result = {
            "input": points,
            "pred": ins_preds,
            "plane": plane_info,
            "rot": plane_info['rotation'],
            "eval": None,
        }
        return exp_result


class UOPModule(PlacementModule):
    def __init__(self, **kwargs):
        self.model = load_model(partial=True, **kwargs)
        
    def inference_gpu(self, points, bw_ratio=0.25):
        points_tensor = torch.from_numpy(points).unsqueeze(0).cuda().float()
        _, logits, embedded, = self.model(points_tensor)
        
        logits = logits.cpu().detach().numpy()
        semantics = np.argmax(logits, axis=-1)
        embedded = embedded.cpu().detach().numpy()
        
        stable_embedded = embedded[0][0 < semantics[0]]
        if 0 < stable_embedded.size:
            try:
                band_width = cluster.estimate_bandwidth(stable_embedded)
                ins_result = MeanShiftEuc(bandwidth=band_width*bw_ratio).fit_predict(stable_embedded)
                ins_result += 1
                
                ins_preds = np.zeros_like(semantics[0])
                ins_preds[semantics[0]>0] = ins_result
                ins_preds[ins_preds>31] = 0
            except:
                print("cluster err")
                ins_preds = np.zeros_like(semantics[0])
        else:
            ins_preds = np.zeros_like(semantics[0])
        
        ins_preds = UOPSIM._reorder_label(ins_preds)
        
        return ins_preds, logits, embedded
    
    def get_stable_placement(self, points):
        ins_preds, logits, embedded = self.inference_gpu(points)
        plane_info = self.get_best_plane_info(points, ins_preds)
        
        if plane_info == {}:
            exp_result = {
                "input": points,
                "pred": ins_preds,
                "plane": plane_info,
                "rot": None,
                "eval": None,
                "logits": logits,
                "embedded": embedded,
            }
        else:    
            exp_result = {
                "input": points,
                "pred": ins_preds,
                "plane": plane_info,
                "rot": plane_info['rotation'],
                "eval": None,
                "logits": logits,
                "embedded": embedded,
            }

        return exp_result


class UOPModuleWhole(UOPModule):
    def __init__(self):
        self.model = load_model(partial=False)


class TrimeshModule(PlacementModule):
    """
    @software{
        trimesh,
        author = {{Dawson-Haggerty et al.}},
        title = {trimesh},
        url = {https://trimsh.org/},
        version = {3.2.0},
        date = {2019-12-8},
    }
    """
    def get_stable_placement(self, points):
        transform, probs, mesh = get_stable_transform(points, return_mesh=True)
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        rot_mat = transform[0][:3, :3]
        exp_result = {
            "input": points,
            "pred": None,
            "plane": None,
            "triangles": triangles,
            "vertices": vertices,
            "rot": rot_mat,
            "eval": None,
        }

        return exp_result


class RansacModule(PlacementModule):
    """
    Yueci Deng, https://github.com/yuecideng/Multiple_Planes_Detection.git
    """
    def __init__(self):
        from placement_module.ransac_module import RemoveNoiseStatistical, DetectMultiPlanes
        self.noise_remover = RemoveNoiseStatistical
        self.plane_detector = DetectMultiPlanes
    
    def get_stable_placement(self, points: np.ndarray):
        pred_points = self.noise_remover(points, nb_neighbors=50, std_ratio=0.5)
        results = self.plane_detector(pred_points, min_ratio=0.05, threshold=0.005, iterations=2000)
    
        # pred
        pred_points = []
        preds = []
    
        best_inlier = 0
        best_plane_info = {}
        instance_idx = 1
        for plane_model, plane_points in results:
            pred_points.append(plane_points)
            preds.append(np.ones(len(plane_points))*instance_idx)
            instance_idx += 1
            
            if best_inlier > len(plane_points):
                continue
                
            best_inlier = len(plane_points)
            plane_center = np.mean(plane_points, axis=0)
            matrix = calculate_placement_transform(plane_model, plane_center, np.zeros(3))
            best_plane_info ={
                "instance_idx": instance_idx,
                "plane_eq": plane_model,
                "plane_center": plane_center,
                "rotation": matrix,
            }
        pred_points = np.concatenate(pred_points, axis=0)
        preds = np.concatenate(preds, axis=0)
        preds += 1
        if best_plane_info == {}:
            exp_result = {
                "input": points,
                "pred_points": pred_points,
                "pred": preds,
                "plane": best_plane_info,
                "rot": None,
                "eval": None,
            }
        else:
            exp_result = {
                    "input": points,
                    "pred_points": pred_points,
                    "pred": preds,
                    "plane": best_plane_info,
                    "rot": best_plane_info['rotation'],
                    "eval": None,
                }
        return exp_result


class PrimitiveFittingModule(PlacementModule):
    
    @staticmethod
    def get_rotation_from_bbox(bbox):
        bbox_points = np.asarray(bbox.get_box_points())
        bbox_center = np.asarray(bbox.get_center())
        planes = [
            (0, 1, 7, 2),
            (0, 2, 5, 3),
            (0, 3, 6, 1),
            (1, 7, 4, 6),
            (2, 5, 4, 7),
            (3, 5, 4, 6),
        ]
        best_normal = None
        best_center = None
        max_area = 0
        for point_idxs in planes:
            square_points = [bbox_points[i] for i in point_idxs]
            plane_center = np.mean(square_points, axis=0)
            normal = plane_center - bbox_center
            area = get_area_from_3points(square_points[:3])
            if max_area < area:
                max_area = area
                best_normal = normal
                best_center = plane_center
        
        rot_mat = calculate_transform_from_normal(best_normal, best_center, bbox_center)

        return rot_mat
    
    def get_stable_placement(self, points: np.ndarray):
        bbox = bbox_fitting_from_points(points)
        bbox_points = np.asarray(bbox.get_box_points())

        rot_mat = self.get_rotation_from_bbox(bbox)
    
        exp_result = {
            "input": points,
            "pred": None,
            "plane": None,
            "bbox_points": bbox_points,
            "rot": rot_mat,
            "eval": None,
        }
        
        return exp_result


available_modules = {
    "gt": GTOracle,
    "trimesh": TrimeshModule,
    "ransac": RansacModule,
    "primitive": PrimitiveFittingModule,
    "uop": UOPModule,
    "uop-whole": UOPModuleWhole,
}


def load_placement_module(module_name, **kwargs):
    
    return available_modules[module_name](**kwargs)

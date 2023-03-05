import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

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


def bbox_fitting_from_points(points: np.ndarray):
  assert len(points) > 3, "There is at most 3 points to fitting plane but got {}".format(len(points))
  
  bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
  
  return bbox
  
  
def get_area_from_3points(points):
  a = points[1] - points[0]
  b = points[2] - points[0]
  return np.linalg.norm(np.cross(a, b))/2


def calculate_placement_transform(plane_model, plane_center, points_centroid):
  # normal vector
  normal = plane_model[:3]
  outer_vec = plane_center - points_centroid
  if np.dot(normal, outer_vec) < 0:
    normal = normal * -1
  
  gravity = np.array([0, 0, -1])
  
  cnt_axis = np.cross(normal, gravity)
  cnt_axis = cnt_axis / (np.linalg.norm(cnt_axis)+1e-9)
  dot_product = np.dot(normal, gravity)/(np.linalg.norm(normal)*np.linalg.norm(gravity))
  angle = np.arccos(dot_product)
  rotation = R.from_rotvec(angle*cnt_axis)

  # rotation, _ = R.align_vectors([normal], [[0, 0, -1]])

  return rotation.as_matrix()

def calculate_transform_from_normal(normal, plane_center, points_centroid):
  # normal vector
  outer_vec = plane_center - points_centroid
  if np.dot(normal, outer_vec) < 0:
    normal = normal * -1
  
  gravity = np.array([0, 0, -1])
  cnt_axis = np.cross(normal, gravity)
  cnt_axis = cnt_axis / np.linalg.norm(cnt_axis)
  dot_product = np.dot(normal, gravity)/(np.linalg.norm(normal)*np.linalg.norm(gravity))
  angle = np.arccos(dot_product)
  rotation = R.from_rotvec(angle*cnt_axis)

  # rotation, _ = R.align_vectors([normal], [[0, 0, -1]])

  return rotation.as_matrix()
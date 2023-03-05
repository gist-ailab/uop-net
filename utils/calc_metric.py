import numpy as np

def calc_transform_diff(previous_mat, current_mat):
    rot_diff = calc_rotation_diff(previous_mat[:3, :3], current_mat[:3, :3])
    trans_diff = calc_translation_diff(previous_mat[:3, 3], current_mat[:3, 3])

    l2norm_diff = calc_l2norm_diff(previous_mat, current_mat)

    return {
        "rotation": rot_diff,
        "translation": trans_diff,
        "l2norm": l2norm_diff
    }

def calc_l2norm_diff(previous_mat, current_mat):
    return np.linalg.norm(previous_mat - current_mat)

def calc_translation_diff(translation_gt, translation_pred):
    """ Computes the distance between the predicted and ground truth translation

    # Arguments
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        The translation distance
    """
    return np.linalg.norm(translation_gt - translation_pred)

def calc_rotation_diff(rotation_gt, rotation_pred):
    """ Calculates the distance between two rotations in degree
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
    # Returns
        the rotation distance in degree
    """  
    rotation_diff = np.dot(rotation_pred, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = (trace - 1.) / 2.
    if trace < -1.:
        trace = -1.
    elif trace > 1.:
        trace = 1.
    angular_distance = np.rad2deg(np.arccos(trace))
    
    return abs(angular_distance)

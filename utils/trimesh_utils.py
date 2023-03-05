import trimesh



def plane_fitting_from_points(points):
    """get plane from points

    Args:
        points (np.ndarray): [(n, 3) target points]

    Returns:
        plane_model: a, b, c, d of equation {ax+by+cz=d}
        inliers: index of inlier points
    """
    assert len(points) > 3, "There is at most 3 points to fitting plane but got {}".format(len(points))
    inlier, normal = trimesh.points.plane_fit(points)
    return normal, inlier


def get_stable_transform(points, return_mesh=False):
    
    pcd = trimesh.points.PointCloud(points)
    tri_mesh = pcd.convex_hull
    
    transform, probs = tri_mesh.compute_stable_poses()
    
    
    if return_mesh:
        mesh = tri_mesh.as_open3d
        return transform, probs, mesh
    else:
        return transform, probs
    
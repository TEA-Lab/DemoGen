import re
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


################################# Camera Calibration ##############################################
# refer to https://gist.github.com/hshi74/edabc1e9bed6ea988a2abd1308e1cc96

ROBOT2CAM_POS = np.array([1.2274124573982026, -0.009193338733170697, 0.3683118830445561])
ROBOT2CAM_QUAT_INITIAL = np.array([0.015873920322366883, -0.18843429010734952, -0.009452363954531973, 0.9819120071477938])

OFFSET_POS=np.array([0.0, -0.01, 0.008])
OFFSET_ORI_X=R.from_euler('x', -1.1, degrees=True)
OFFSET_ORI_Y=R.from_euler('y', 1.1, degrees=True)
OFFSET_ORI_Z=R.from_euler('z', -1.6, degrees=True)

ROBOT2CAM_POS = ROBOT2CAM_POS + OFFSET_POS
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X * OFFSET_ORI_Y * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()

robot2cam_mat = np.eye(4)
robot2cam_mat[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
robot2cam_mat[:3, 3] = ROBOT2CAM_POS


REALSENSE_SCALE = 0.0002500000118743628
fx = 1354.796875
fy = 1354.6197509765625
cx = 986.0130004882812
cy = 548.7333374023438
intrinsic_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])


T_link2viz = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])

transform_realsense_util = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
image_size = (1920, 1080)
##############################################



def inverse_extrinsic_matrix(matrix):
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

def restore_original_pcd(transformed_points):
    xyz = transformed_points[:, :3]
    rgb = transformed_points[:, 3:]

    robot2cam_extrinsic_matrix = np.eye(4)
    robot2cam_extrinsic_matrix[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
    robot2cam_extrinsic_matrix[:3, 3] = ROBOT2CAM_POS
    
    cam_T_robot = inverse_extrinsic_matrix(robot2cam_extrinsic_matrix)
    viz_T_link = inverse_extrinsic_matrix(T_link2viz)
    inverse_trans = inverse_extrinsic_matrix(transform_realsense_util)

    homogeneous_points = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

    restored_points = homogeneous_points.T
    restored_points = cam_T_robot @ restored_points
    restored_points = viz_T_link @ restored_points
    restored_points = inverse_trans @ restored_points
    restored_points = restored_points.T
    
    restored_xyz = restored_points[:, :3]
    restored_xyz /= REALSENSE_SCALE
    restored_points = np.hstack((restored_xyz, rgb))

    return restored_points

def project_points_to_image(point_cloud, K, R=np.eye(3), T=np.zeros(3)):
    points_3d = point_cloud[:, :3]
    points_3d = (R @ points_3d.T).T + T
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    K_proj = np.hstack((K, np.zeros((3, 1))))
    points_2d_homogeneous = (K_proj @ points_3d_homogeneous.T).T
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]

    return points_2d


def filter_points_by_mask(points, mask, intrinsic_matrix, image_size):
    projected_points = project_points_to_image(points, intrinsic_matrix, R=np.eye(3), T=np.zeros(3))
    pixel_coords = np.floor(projected_points).astype(int) 
    
    valid_points = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_size[0]) & \
                   (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_size[1])
    
    mask_values = mask[pixel_coords[valid_points, 1], pixel_coords[valid_points, 0]]
    
    final_mask = np.zeros(len(points), dtype=bool)
    final_mask[valid_points] = mask_values
    
    filtered_points = points[final_mask]
    
    return filtered_points

def trans_pcd(points):
    robot2cam_extrinsic_matrix = np.eye(4)
    robot2cam_extrinsic_matrix[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
    robot2cam_extrinsic_matrix[:3, 3] = ROBOT2CAM_POS
    # scale
    points_xyz = points[..., :3] * REALSENSE_SCALE
    point_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
    point_homogeneous = transform_realsense_util @ point_homogeneous.T
    point_homogeneous = T_link2viz @ point_homogeneous
    point_homogeneous = robot2cam_extrinsic_matrix @ point_homogeneous
    point_homogeneous = point_homogeneous.T

    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
    return points

def restore_and_filter_pcd(pcd_robot, mask, intrinsic_matrix=intrinsic_matrix, image_size=image_size):
    pcd_cam = restore_original_pcd(pcd_robot)
    filtered_points = filter_points_by_mask(pcd_cam, mask, intrinsic_matrix, image_size)
    filtered_points = trans_pcd(filtered_points)
    return filtered_points

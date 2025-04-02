import argparse
import os
import numpy as np
import zarr
import pickle
from termcolor import cprint

from scipy.spatial.transform import Rotation as R
import fpsample
from sklearn.cluster import DBSCAN
from time import time
import pcd_visualizer
from dataclasses import dataclass, field
from typing import List
from realsense_camera import RealSense_Camera


################################# Camera Calibration ##############################################
# refer to https://gist.github.com/hshi74/edabc1e9bed6ea988a2abd1308e1cc96

OFFSET=np.array([0.0, 0.0, -0.035])
ROBOT2CAM_POS = np.array([1.2274124573982026, -0.009193338733170697, 0.3683118830445561]) + OFFSET
ROBOT2CAM_QUAT_INITIAL = np.array(
    [0.015873920322366883, -0.18843429010734952, -0.009452363954531973, 0.9819120071477938]
)
OFFSET_ORI_X=R.from_euler('x', -1.2, degrees=True)
ori = R.from_quat(ROBOT2CAM_QUAT_INITIAL) * OFFSET_ORI_X
OFFSET_ORI_Y=R.from_euler('y', 10, degrees=True)
ori = ori * OFFSET_ORI_Y
OFFSET_ORI_Z=R.from_euler('z', 0, degrees=True)
ori = ori * OFFSET_ORI_Z
ROBOT2CAM_QUAT = ori.as_quat()

REALSENSE_SCALE = 0.0002500000118743628
quat = [-0.491, 0.495, -0.505, 0.509]
pos = [0.004, 0.001, 0.014]
# transformation of color link (child) in the robot base frame (parent)
T_link2color = np.concatenate((np.concatenate((R.from_quat(quat).as_matrix(), np.array([pos]).T), axis=1), [[0, 0, 0, 1]]))
T_link2viz = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
transform_realsense_util = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
###################################################################################################


################################# Hyperparameters for pcd_process ##################################
@dataclass
class PCDProcConfig:
    random_drop_points: int
    outlier_distance: float
    outlier_count: int
    n_points: int
    work_space: List[List[float]]

pcd_config = PCDProcConfig(
    random_drop_points=5000,
    outlier_distance=0.012,
    outlier_count=50,
    n_points=1024,
    work_space=[
        [0.2, 0.8],
        [-0.66, 0.6],
        [0.005, 0.45]
    ])
###################################################################################################


def preprocess_point_cloud(points, cfg=pcd_config, debug=False):
    points = pcd_crop(points, cfg, debug)
    points = pcd_cluster(points, cfg, debug)
    return points

def pcd_crop(points, cfg=pcd_config, debug=False):
    WORK_SPACE = cfg.work_space
    
    robot2cam_extrinsic_matrix = np.eye(4)
    robot2cam_extrinsic_matrix[:3, :3] = R.from_quat(ROBOT2CAM_QUAT).as_matrix()
    robot2cam_extrinsic_matrix[:3, 3] = ROBOT2CAM_POS

    points_xyz = points[..., :3] * REALSENSE_SCALE
    point_homogeneous = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
    point_homogeneous = T_link2viz @ point_homogeneous.T
    point_homogeneous = robot2cam_extrinsic_matrix @ point_homogeneous
    point_homogeneous = point_homogeneous.T

    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz

    if debug:
        pcd_visualizer.visualize_pointcloud(points)
    
    # Crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    
    if debug:
        pcd_visualizer.visualize_pointcloud(points)
    
    return points


def pcd_cluster(points, cfg=pcd_config, debug = False):
    RANDOM_DROP_POINTS = cfg.random_drop_points
    OUTLIER_DISTANCE = cfg.outlier_distance
    OUTLIER_COUNT = cfg.outlier_count
    N_POINTS = cfg.n_points

    # Randomly drop points
    points = points[np.random.choice(points.shape[0], RANDOM_DROP_POINTS, replace=False)]
    points_xyz = points[..., :3]

    # DBSCAN clustering
    bdscan = DBSCAN(eps=OUTLIER_DISTANCE, min_samples=10)
    labels = bdscan.fit_predict(points_xyz)

    # Then get out of the cluster with less than OUTLIER points or noise
    unique_labels, counts = np.unique(labels, return_counts=True)
    outlier_labels = unique_labels[counts < OUTLIER_COUNT]
    if -1 not in outlier_labels:
        outlier_labels = np.append(outlier_labels, -1)

    points = points[~np.isin(labels, outlier_labels) ]
    points_xyz = points[..., :3]

    if debug:
        pcd_visualizer.visualize_pointcloud(points)

    # FPS sampling
    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, N_POINTS, h=3)
    points = points[sample_indices]

    if debug:
        pcd_visualizer.visualize_pointcloud(points)

    return points


if __name__ == "__main__":
    id = "f0211830"
    realsense_camera = RealSense_Camera(type="L515", id=id)
    realsense_camera.prepare()
    point_cloud, rgbd_frame = realsense_camera.get_frame()

    preprocess_point_cloud(points=point_cloud, debug=False)
    # save to npy
    np.save("./data/moon.npy", point_cloud)
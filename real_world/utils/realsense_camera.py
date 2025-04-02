
import numpy as np
import math
import time
import cv2
import pyrealsense2 as rs

# PIXEL_W, PIXEL_H = 640, 480 # default
DEPTH_PIXEL_W, DEPTH_PIXEL_H = 640, 480 #320, 240
COLOR_PIXEL_W, COLOR_PIXEL_H = 640, 480 #640, 360


class CameraInfo():
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed




class RealSense_Camera:
    def __init__(self, type="L515", id= None):
        self.pc = rs.pointcloud()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id!=None and type=="L515": # D435 doesn't need this
            self.config.enable_device(id)
        self.config.enable_stream(rs.stream.depth, DEPTH_PIXEL_W, DEPTH_PIXEL_H, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, COLOR_PIXEL_W, COLOR_PIXEL_H, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def prepare(self):
        for fid in range(50):
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

    def get_frame(self, remove_bg = False):
        frames = self.pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # cv2.imshow("color_image", color_image)
        
        aligned_frames = self.align.process(frames)
        profile = aligned_frames.get_profile()
        intrinsics = profile.as_video_stream_profile().get_intrinsics()
        camera = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        # print(camera)
        depth_frame = aligned_frames.get_depth_frame() 
        
        depth_image = np.asanyarray(depth_frame.get_data())
        
        color_image[:, :, [0,2]] = color_image[:,:,[2,0]]
        transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        
        if not remove_bg:
            point_xyz = create_point_cloud_from_depth_image(depth_image, camera, organized=False)
            point_xyz = transform_point_cloud(point_xyz, transform)
            rgbd_frame = np.concatenate([color_image, np.expand_dims(depth_image, axis=-1)], axis = -1)
            point_color = color_image.reshape(-1, 3)


        else:
            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            point_xyz = create_point_cloud_from_depth_image(depth_image, camera, organized=False)
            transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            point_xyz = transform_point_cloud(point_xyz, transform)
            point_color = bg_removed.reshape([-1,3])
            point_color = transform_point_cloud(point_color, transform)
            rgbd_frame = np.concatenate([bg_removed, np.expand_dims(depth_image, axis=-1)], axis = -1)

        point_cloud = np.concatenate([point_xyz, point_color], axis = 1)

        return point_cloud, rgbd_frame


        

  
from polymetis import RobotInterface
from oyhand_node import OYhandNode
import numpy as np
import torch
from gym import spaces
import time
import cv2
from scipy.spatial.transform import Rotation as R,Slerp

from realsense_camera import RealSense_Camera
from pcd_process import preprocess_point_cloud, pcd_crop, pcd_cluster

from pcd_visualizer import visualize_pointcloud

CAMERA_ID = 'f0211830'

ARM_HOME = np.array([0.49, 0.01, 0.44, 3.14, 0, 2.4])
# ARM_HOME = np.array([0.54, -0.35, 0.29, 3.14, 0, 2.4])  # left

class PandaOYhandEnv:
    def __init__(self, camera="L515"):
        
        self.arm_action_dim = 6
        self.hand_action_dim = 6
        self.num_points = 1024
        self.arm_home = ARM_HOME
        
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.arm_action_dim + self.hand_action_dim,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, 84, 84),
                dtype=np.float32
            ),
            
            'depth': spaces.Box(
                low=0,
                high=1,
                shape=(84, 84),
                dtype=np.float32
            ),
            
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),

        })
        
        self.arm = RobotInterface(ip_address="172.16.0.11")
        self.arm.start_cartesian_impedance()
        self.hand = OYhandNode()

        if camera is not None:
            self.realsense_camera = RealSense_Camera(type=camera, id=CAMERA_ID)
            self.realsense_camera.prepare()
        
        # self.go_home()
    
    def start(self):
        print("Force reader started")
        # Start the UDP receiver
        #self.force_sensor.start()

    def stop(self):
        print("Stop receiver...")
        # Stop the UDP receiver of the FSR
        #self.force_sensor.stop()  
        
    def go_home(self, arm_home=ARM_HOME):
        self.arm.move_to_ee_pose(torch.Tensor(arm_home[:3]), torch.Tensor(R.from_euler('XYZ', arm_home[3:]).as_quat()), 3.0)
        self.hand.open()
        print('Move to home!')
        time.sleep(0.1)
        self.arm = RobotInterface(ip_address="172.16.0.11")
        self.arm.start_cartesian_impedance()
        self.latest_arm_action = arm_home

    def step(self, action):
        """
        arm action: x, y, z, euler-z, euler-y, euler-x
        gripper action: -1 or 1
        """
        # print("env step action:", action)
        assert len(action) == self.arm_action_dim + self.hand_action_dim
        arm_action = action[:self.arm_action_dim]
        hand_action = action[self.arm_action_dim:]
        self.move_robot(arm_action, hand_action)
        obs_dict = self.get_obs()
        return obs_dict, 0, False, {}

    def render(self, mode=None):
        point_cloud, rgbd_frame = self.get_point_cloud_with_image()
        rgb = rgbd_frame[:, :, :3].astype(np.uint8)
        return rgb
        
    def get_point_cloud_with_image(self):
        point_cloud, rgbd_frame = self.realsense_camera.get_frame()
        return point_cloud, rgbd_frame

    def get_robot_state(self):
        state_arm_ee_pos, state_arm_ee_quat = self.arm.get_ee_pose()
        state_arm_ee_pos = state_arm_ee_pos.numpy()
        state_arm_ee_quat = state_arm_ee_quat.numpy()
        state_arm_ee_euler = R.from_quat(state_arm_ee_quat).as_euler('XYZ')
        state_arm_ee = np.concatenate([state_arm_ee_pos, state_arm_ee_euler])
        state_hand_joint = self.hand.read_pos()
        # state_agent_pos = np.concatenate([state_arm_ee, state_hand_joint])  # 6 + 16
        
        state_arm_ee=state_arm_ee
        state_agent_pos = np.concatenate([state_arm_ee, state_hand_joint])
        return state_agent_pos
    
    def get_rgb_image(self):
        _, rgbd_frame = self.get_point_cloud_with_image()
        rgb = rgbd_frame[:, :, :3]
        return rgb
    
    def get_hand_state(self):
        return self.hand.read_pos()
    
    def move_hand(self, hand_action):
        self.hand.set_pos(hand_action)
    
    def get_obs(self, smooth=False):
        robot_state = self.get_robot_state()
        point_cloud, rgbd_frame = self.get_point_cloud_with_image()
        #print(point_cloud.shape)
        # point_cloud = preprocess_point_cloud(points=point_cloud)
        # visualize_pointcloud(point_cloud)
        
        point_cloud = pcd_crop(point_cloud)
        
        # visualize_pointcloud(point_cloud)
        
        if smooth:
            self.update_arm(self.latest_arm_action)
        point_cloud = pcd_cluster(point_cloud)
        
        # visualize_pointcloud(point_cloud)
        
        rgb = rgbd_frame[:, :, :3]
        depth = rgbd_frame[:, :, -1]
        obs_dict = {
            'point_cloud': point_cloud,
            'image': rgb,
            'depth': depth,
            'agent_pos': robot_state,
        }
        if smooth:
            self.update_arm(self.latest_arm_action)
        return obs_dict

    def reset(self):
        self.go_home()
        return self.get_obs()
    
    def update_arm(self, arm_action):
        self.arm.update_desired_ee_pose(torch.Tensor(arm_action[:3]), torch.Tensor(R.from_euler('XYZ', arm_action[3:]).as_quat()))
    
        
    def interpolate_robot(self,action):
        N_INTERPOLATE = 30
        desired_pos_action=torch.Tensor(action[:3])
        desired_rot_action=torch.Tensor(R.from_euler('XYZ', action[3:]).as_quat())
        
        init_action=self.arm.get_ee_pose()
#        rot_start=R.from_quat(np.array(init_action[0]))
       # rot_end=R.from_quat(np.array(desired_rot_action))
        key_times=[0,1]
        rotations=R.from_quat([np.array(init_action[1]),np.array(desired_rot_action)])
        slerp=Slerp(key_times,rotations)
        t=np.linspace(0, 1, num=N_INTERPOLATE, endpoint=False) + 1. / N_INTERPOLATE
        interpolated_rotations=slerp(t)
        
        
        for i,rotation in zip(t,interpolated_rotations):
            
            inte_pos_action = init_action[0] + i * (desired_pos_action - init_action[0])
            inte_rot_action = torch.Tensor(rotation.as_quat())
            #inte_rot_action=init_action[1]+i/interpolate_number*(desired_rot_action-init_action[1])
            self.arm.update_desired_ee_pose(inte_pos_action,inte_rot_action)
            time.sleep(0.1 / N_INTERPOLATE)
            
    def move_robot(self, arm_action, hand_action, verbose=False, use_relative = True):
        self.hand.set_pos(hand_action)
        # self.update_arm(arm_action)
        self.interpolate_robot(arm_action)
        self.latest_arm_action = arm_action

 
if __name__ == '__main__':
    env = PandaOYhandEnv()
    
    # for _ in range(100):
    #     # print(env.force_sensor.get_most_recent_pressure())
        
    #     rgb_img = env.get_rgb_image()
    #     print(rgb_img.shape, rgb_img.max(), rgb_img.min())
    #     # cv2.imshow("rgb_image", rgb_img)
    #     # save
    #     cv2.imwrite("rgb_image.png", rgb_img)
        
    #     time.sleep(0.1)
    #     # print(env.get_hand_state())
    
    # thumb bend, index, middle, ring, little, thumb rot
    hand_action_seq = [
        [0.64, 1.57, 1.57, 1.57, 1.57, 0],
        [0.2, 3.14, 3.14, 3.14, 3.14, 1.57],
        [0.64, 1.57, 3.14, 3.14, 3.14, 1.57],
        [0.64, 1.57, 1.57, 3.14, 3.14, 0],
        [0.64, 1.57, 1.57, 1.57, 3.14, 0],
        [0.64, 1.57, 1.57, 1.57, 1.57, 0],
        [0.64, 3.14, 3.14, 3.14, 3.14, 0],
        [0.64, 3.14, 3.14, 3.14, 3.14, 0],
    ]
    
    delta_arm_action = np.array([0.01, 0.01, 0.01, 0, 0, 0])
    
    for t, hand_action in enumerate(hand_action_seq):
        arm_action = ARM_HOME + t * delta_arm_action
        action = np.concatenate([arm_action, hand_action])
        env.step(action)
        
        # env.move_hand(hand_action)
        
        # rgb = env.get_rgb_image()
        
        # time.sleep(1)
    # cv2.destroyAllWindows()

    
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from diffusion_policies.gym_util.mjpc_wrapper import point_cloud_sampling

from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robosuite.controllers import load_composite_controller_config
from pcd_visualizer import visualize_pointcloud


TASK_BOUDNS = {
    # x_min, y_min, z_min, x_max, y_max, z_max
    'Coffee': [-0.4, -0.5, 0.810, 0.19, 0.5, 1.15],
    'Kitchen': [-0.5, -0.5, 0.912, 0.2, 0.5, 1.3],
    'HammerCleanup': [-0.4, -0.5, 0.91, 0.19, 0.5, 1.3],
    # 'MugCleanup': [-0.4, -0.5, 0.91, 0.19, 0.5, 1.3],
    'ThreePieceAssembly': [-0.4, -0.5, 0.81, 0.36, 0.5, 1.3],
}

SINGLE_CEMERA_NAME = {
    'Coffee': 'myfrontlow',
    'Kitchen': 'myfronthigh',
    'HammerCleanup': 'myfronthigh',
    'ThreePieceAssembly': 'myfrontlow',
}

# MULTI_VIEW_CAMS = ['birdview', 'agentview', 'left', 'right', 'robot0_eye_in_hand']
MULTI_VIEW_CAMS = ['left', 'right', 'agentview']
SINGLE_VIEW_CAMS = ['myfront']
MULTI_VIEW_NUM_POINTS = 1024
SINGLE_VIEW_NUM_POINTS = 512
USE_CROP = True
N_CONTROL_STEPS = 5     # 2


class Robosuite3DEnv(gym.Env):
    def __init__(self, source_demo_path,
                 multi_view=False,
                 cam_width=128,
                 cam_height=128,
                 render=False,
                 render_cam="myfront",
                 support_osc_control=False,
                ):
        if multi_view:
            raise NotImplementedError("Multi-view not supported.")
        else:
            self.n_points = SINGLE_VIEW_NUM_POINTS
        

        dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[],),)
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
        env_meta = FileUtils.get_env_metadata_from_dataset(source_demo_path)
        print("EnvUtils.is_robosuite_env(env_meta)", EnvUtils.is_robosuite_env(env_meta))
        if not support_osc_control:
            controller_config = load_composite_controller_config(
                controller='WHOLE_BODY_MINK_IK',
                robot="Panda",
            )
            env_meta["env_kwargs"]["controller_configs"] = controller_config

        self.task_name = env_meta["env_name"].split("_")[0]
        if self.task_name not in TASK_BOUDNS.keys():
            raise ValueError(f"Task {self.task_name} not supported.")
        env_meta["env_name"] = self.task_name + "_D0"

        self.cam_width = cam_width
        self.cam_height = cam_height
        env_meta["env_kwargs"]["camera_heights"] = cam_height
        env_meta["env_kwargs"]["camera_widths"] = cam_width
        env_meta["env_kwargs"]["render_camera"] = render_cam


        if multi_view:
            self.cam_names = MULTI_VIEW_CAMS
            self.n_points = MULTI_VIEW_NUM_POINTS
        else:
            self.cam_names = [SINGLE_CEMERA_NAME[self.task_name]]
            env_meta["env_kwargs"]["camera_names"] = self.cam_names
        
        self.env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=render,
                                                use_image_obs=True, use_depth_obs=True)
        
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "point_cloud": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_points, 6), dtype=np.float32),
        })
    
    @staticmethod
    def _convert_state_quat_to_action_rotvec(state_quat):
        """
        Handle confusing rotation transform.

        Args:
            state_quat (np.array): 4-element quaternion representing the state

        Returns:
            action_rotvec (np.array): 3-element rotation vector representing the action
        """
        state_mat = R.from_quat(state_quat).as_matrix()
        T = R.from_rotvec([0, 0, -1.5707]).as_matrix()
        action_mat = state_mat @ T
        return R.from_matrix(action_mat).as_rotvec()

    @staticmethod
    def _pcd_crop(pcd, bounds):
        mask = np.all(pcd[:, :3] > bounds[:3], axis=1) & np.all(pcd[:, :3] < bounds[3:], axis=1)
        return pcd[mask]

    def process_obs_dict(self, obs_dict):
        keys = obs_dict.keys()
        # print("keys:", keys)
        processed_obs = {
            "agent_pos": np.zeros(7),
            "point_cloud": np.zeros((self.n_points, 6)),
        }

        # process robot state
        robot_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        if all([k in keys for k in robot_keys]):
            pos = obs_dict["robot0_eef_pos"]
            rot = self._convert_state_quat_to_action_rotvec(obs_dict["robot0_eef_quat"])
            gripper = obs_dict["robot0_gripper_qpos"][0] - obs_dict["robot0_gripper_qpos"][1]
            robot_state = np.concatenate([pos, rot, [gripper]])
            processed_obs["agent_pos"] = robot_state

        # process point cloud
        pcd_list = []
        for cam_name in self.cam_names:
            rgb_name = cam_name + "_image"
            depth_name = cam_name + "_depth"
            if all([k in keys for k in [rgb_name, depth_name]]):
                # print("cam_name:", cam_name)
                rgb = obs_dict[rgb_name]
                depth = obs_dict[depth_name]
                pcd = self.get_pcd_from_rgbd(cam_name, rgb, depth)
                pcd_list.append(pcd)
            else:
                break
        if len(pcd_list) > 0:
            pcd = np.concatenate(pcd_list, axis=0)
            if USE_CROP:
                pcd = self._pcd_crop(pcd, TASK_BOUDNS[self.task_name])
            pcd = point_cloud_sampling(pcd, self.n_points)
            processed_obs["point_cloud"] = pcd

            # import pcd_visualizer
            # pcd_visualizer.visualize_pointcloud(pcd)
        
        return processed_obs

    def get_pcd_from_rgbd(self, cam_name, rgb, depth):
        def verticalFlip(img):
            return np.flip(img, axis=0)
        
        def get_o3d_cammat():
            cam_mat = self.env.get_camera_intrinsic_matrix(cam_name, self.cam_width, self.cam_height)
            cx = cam_mat[0,2]
            fx = cam_mat[0,0]
            cy = cam_mat[1,2]
            fy = cam_mat[1,1]
            return o3d.camera.PinholeCameraIntrinsic(self.cam_width, self.cam_height, fx, fy, cx, cy)
        
        rgb = verticalFlip(rgb)
        depth = self.env.get_real_depth_map(verticalFlip(depth))
        o3d_cammat = get_o3d_cammat()
        o3d_depth = o3d.geometry.Image(depth)
        o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_cammat)
        world_T_cam = self.env.get_camera_extrinsic_matrix(cam_name)
        o3d_pcd.transform(world_T_cam)
        points = np.asarray(o3d_pcd.points)
        colors = rgb.reshape(-1, 3)
        pcd = np.concatenate([points, colors], axis=1)
        return pcd

    def reset(self):
        obs_dict = self.env.reset()
        return self.process_obs_dict(obs_dict)
    
    def reset_to(self, state):
        return self.env.reset_to(state)
    
    def step(self, action):
        for _ in range(N_CONTROL_STEPS):
            obs_dict, r, done, info = self.env.step(action)
        return self.process_obs_dict(obs_dict), r, done, info
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def check_success(self):
        return self.env.env._check_success()
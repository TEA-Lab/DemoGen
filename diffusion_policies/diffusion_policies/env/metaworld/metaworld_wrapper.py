import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import cv2

from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policies.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policies.gym_util.mjpc_wrapper import point_cloud_sampling


TASK_BOUDNS = {
    # x_min, y_min, z_min, x_max, y_max, z_max (0.4)
    'default': [-1, 0.25, -0.001, 1, 1.3, 0.37],
}

# TASK_BOUDNS = {
#     # x_min, y_min, z_min, x_max, y_max, z_max (0.4)
#     'default': [-1, 0.35, -0.001, 1, 1.3, 0.3],
# }


# cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
# WRIST_CAMERA_NAME = 'gripperPOV50'    # wrist camera
WRIST_CAMERA_NAME = 'gripper435'    # wrist camera
WRIST_2_CAMERA_NAME = 'gripper120'    # wrist camera, other side
MAIN_CAMERA_NAME = 'front30'    # third-person view, front435
SIDE_CAMERA_NAME = 'corner'
IMG_SIZE = 224
NUM_POINTS = 512    # 512
DEPTH_CAMERA_NAME = 'depth'
DEPTH_IMG_SIZE = 224
DEPTH_WIDTH = 120
DEPTH_HEIGHT = 100
N_SIM_STEPS = 5     # make EE movement more accurate
MAX_EP_LENGTH = 200
COLLISION_THRESHOLD = 0.01


class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 reset_mode="default",
                 image_obs_only=False,
                 state_obs_only=False
                 ):
        # print("MetaWorldEnv init", reset_mode)
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False # this is important for domain randomization

        self.cam_id = self.env.sim.model.camera_name2id(MAIN_CAMERA_NAME)
        self.pc_offset = self.env.sim.model.cam_pos[self.cam_id]
        self.pc_transform = self.env.sim.model.cam_mat0[self.cam_id].reshape(3, 3)
        self.pc_scale = np.array([1, 1, 1])

        # print("after super", reset_mode)

        self.reset_mode = reset_mode
        self.env.reset_mode = reset_mode
        self.image_obs_only = image_obs_only
        self.state_obs_only = state_obs_only
        
        ####################################################################
        # BUG: Delete this. Otherwise wrist camera observation will crash.
        # !!! this is very important, otherwise the point cloud will be wrong.
        # self.env.sim.model.vis.map.znear = 0.1
        # self.env.sim.model.vis.map.zfar = 1.5
        #####################################################################
        
        self.device = device
        self.device_id = int(device.split(":")[-1])

        # self.device_id = 0
        # cprint("Using device 0 for env render", 'light_red')
        
        # set the device of mujoco simulation

        
        self.image_size = IMG_SIZE
        self.depth_image_size = DEPTH_IMG_SIZE
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=[DEPTH_CAMERA_NAME], img_width=DEPTH_WIDTH, img_height=DEPTH_HEIGHT)
        self.use_point_crop = use_point_crop
        # cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        # self.num_points = num_points # 512
        self.num_points = NUM_POINTS
        
        x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
        
        # self.episode_length = self.env.max_path_length # this is 500
        self.episode_length = self._max_episode_steps = MAX_EP_LENGTH
        
        
        self.action_space = self.env.action_space
        
        # cprint("[MetaWorldEnv] action_space: {}".format(self.env.action_space.shape), "yellow")

        self.obs_sensor_dim = self.get_robot_state().shape[0]

        
    
        self.observation_space = spaces.Dict({
            'main_img': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'wrist_img': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'wrist2_img': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'side_img': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.depth_image_size, self.depth_image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
            'cheat_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(10, ),
                dtype=np.float32
            ),
        })

    @staticmethod
    def cheat_state_extractor(raw_state):
        ee = raw_state[:4]
        object_pos = raw_state[4:7]
        target_pos = raw_state[-3:]
        return np.concatenate([ee, object_pos, target_pos])

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        # finger_gap = finger_right - finger_left
        finger_gap = [finger_left[1] - finger_right[1]]
        # print("finger_gap:", finger_gap)
        # return np.concatenate([eef_pos, finger_right, finger_left])
        return np.concatenate([eef_pos, finger_gap])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        # os.system("unset LD_PRELOAD")
        main_img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name=MAIN_CAMERA_NAME, device_id=self.device_id)
    

        # main_img = self.env.sim.render(width=112, height=224, camera_name=MAIN_CAMERA_NAME, device_id=self.device_id)
        # main_img = cv2.resize(main_img, (self.image_size, self.image_size))

        wrist_img = self.env.sim.render(width=320, height=640, camera_name=WRIST_CAMERA_NAME, device_id=self.device_id)
        wrist_img = cv2.resize(wrist_img, (self.image_size, self.image_size))
        # print(wrist_img.shape)


        wrist2_img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name=WRIST_2_CAMERA_NAME, device_id=self.device_id)
        # side_img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name=SIDE_CAMERA_NAME, device_id=self.device_id)
        
        side_img = self.env.sim.render(width=DEPTH_WIDTH, height=DEPTH_HEIGHT, camera_name=DEPTH_CAMERA_NAME, device_id=self.device_id)
        side_img = cv2.resize(side_img, (self.image_size, self.image_size))
        
        return main_img, wrist_img, wrist2_img, side_img

    def render_high_res(self, resolution=1024):
        main_img = self.env.sim.render(width=resolution, height=resolution, camera_name=MAIN_CAMERA_NAME, device_id=self.device_id)
        wrist_img = self.env.sim.render(width=resolution, height=resolution, camera_name=WRIST_CAMERA_NAME, device_id=self.device_id)
        wrist2_img = self.env.sim.render(width=resolution, height=resolution, camera_name=WRIST_2_CAMERA_NAME, device_id=self.device_id)
        side_img = self.env.sim.render(width=resolution, height=resolution, camera_name=SIDE_CAMERA_NAME, device_id=self.device_id)
        return main_img, wrist_img, wrist2_img, side_img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        # do transform, scale, offset, and crop
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        # import pcd_visualizer; pcd_visualizer.visualize_pointcloud(point_cloud)
        # import ipdb; ipdb.set_trace()
            
        # self.use_point_crop = False
        if self.use_point_crop:
            # if self.min_bound is not None:
            #     mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
            #     point_cloud = point_cloud[mask]
            # if self.max_bound is not None:
            #     mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
            #     point_cloud = point_cloud[mask]
        
            point_cloud = point_cloud[
            (point_cloud[:, 0] >= self.min_bound[0]) & (point_cloud[:, 0] <= self.max_bound[0]) &
            (point_cloud[:, 1] >= self.min_bound[1]) & (point_cloud[:, 1] <= self.max_bound[1]) &
            (point_cloud[:, 2] >= self.min_bound[2]) & (point_cloud[:, 2] <= self.max_bound[2])]

        
        # print("y_min before fps:", min(point_cloud[:, 1]))

        # import pcd_visualizer; pcd_visualizer.visualize_pointcloud(point_cloud)
        # import ipdb; ipdb.set_trace()

        # fps_start = time.time()
        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps', self.device)

        # print("pcd after fps:", point_cloud[:3])
        # print("y_min after fps:", min(point_cloud[:, 1]))


        # print("FPS sampling time: ", time.time() - fps_start)
        
        # import pcd_visualizer; pcd_visualizer.visualize_pointcloud(point_cloud)
        # import ipdb; ipdb.set_trace()
        
        depth = depth[::-1] # flip vertically
        
        return point_cloud, depth
        
 
    def get_visual_obs(self):
        robot_state = self.get_robot_state()
        if not self.state_obs_only:
            main_img, wrist_img, wrist2_img, side_img = self.get_rgb()
        else:
            main_img = np.zeros((3, self.image_size, self.image_size))
            wrist_img = np.zeros((3, self.image_size, self.image_size))
            wrist2_img = np.zeros((3, self.image_size, self.image_size))
            side_img = np.zeros((3, self.image_size, self.image_size))

        if main_img.shape[0] != 3:  # make channel first
            main_img = main_img.transpose(2, 0, 1)
        if wrist_img.shape[0] != 3:  # make channel first
            wrist_img = wrist_img.transpose(2, 0, 1)
        if wrist2_img.shape[0] != 3:  # make channel first
            wrist2_img = wrist2_img.transpose(2, 0, 1)
        if side_img.shape[0] != 3:  # make channel first
            side_img = side_img.transpose(2, 0, 1)
        

        # print("finger gap", robot_state[7] - robot_state[4])

        if not self.image_obs_only and not self.state_obs_only:
            point_cloud, depth = self.get_point_cloud()
        else:
            point_cloud = np.zeros((self.num_points, 3))
            depth = np.zeros((self.depth_image_size, self.depth_image_size))
        
        obs_dict = {
            'main_img': main_img,
            'wrist_img': wrist_img,
            'wrist2_img': wrist2_img,
            'side_img': side_img,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        action = np.clip(action, -1, 1)

        # import ipdb; ipdb.set_trace()

        ee_pos_before = self.get_robot_state()[:3]
        ee_pos_desire = ee_pos_before + action[:3] / 100
        vfunc = np.vectorize("{:.2e}".format)
        # print("action scaled: ", vfunc(action[:3] / 100))
        # cprint(f"ee_pos_desire: {ee_pos_desire}", "light_blue")

        full_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1
        # print("self.cur_step: ", self.cur_step)

        blank_action = action.copy()
        blank_action[:3] = 0
        # print("blank_action: ", blank_action)
        for _ in range(N_SIM_STEPS):
            self.env.step(blank_action)

        ee_pos_after = self.get_robot_state()[:3]
        mocap_target = self.env.data.mocap_pos

        mocap_pos_diff = mocap_target - ee_pos_after
        # print(mocap_pos_diff)

        if mocap_pos_diff[0][-1] < - COLLISION_THRESHOLD:
            # cprint(f"Collision occurs? mocap_pos_diff: {vfunc(mocap_pos_diff)}", 'red')
            # cprint(f"mocap_target: {vfunc(mocap_target)}", 'blue')
            env_info['collision'] = True
            done = True
        else:
            env_info['collision'] = False
            

        # print("ee_pos_after: ", ee_pos_after)
        # ee_pos_diff = ee_pos_desire - ee_pos_after
        
        delta_action = ee_pos_after - ee_pos_desire
        # cprint(f"mocap_pos_diff: {vfunc(mocap_pos_diff)}", 'red')
        # cprint(f"delta_action: {vfunc(delta_action)}", "blue")

        obs_dict = self.get_visual_obs()
        obs_dict['full_state'] = full_state
        obs_dict['cheat_state'] = self.cheat_state_extractor(full_state)


        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self, config_idx=None):
        self.env.reset()
        full_state = self.env.reset_model(config_idx)
        # if config_idx is not None:
        #     # print("reset with config index: ", config_idx)
        #     full_state = self.env.reset_model_index(config_idx)  # deterministic reset model with config index
        #     # print("env._target_pos before env.reset:", self.env.get_target_pos())
        # else:
        #     full_state = self.env.reset_model()
        
        if full_state is None:  # handling skipped demo
            return None

        # raw_obs = self.env.reset()
        # print("env._target_pos after env.reset:", self.env.get_target_pos())
        self.cur_step = 0

        # make the EE movement more accurate
        for _ in range(N_SIM_STEPS * 5):
            self.env.step(np.zeros(4))
            
        obs_dict = self.get_visual_obs()
        obs_dict['full_state'] = full_state
        obs_dict['cheat_state'] = self.cheat_state_extractor(full_state)

        return obs_dict
    
    def reset_to_state(self, object_states):
        return NotImplementedError

    def seed(self, seed=None):
        # self.env.seed(seed)
        pass

    def set_seed(self, seed=None):
        # self.env.seed(seed)
        pass

    def render(self, mode='rgb_array'):
        main_img, wrist_img, wrist2_img, side_img = self.get_rgb()
        return main_img, wrist_img, wrist2_img, side_img

    def close(self):
        pass

    def get_target_pos(self):
        return self.env._target_pos
    
    def get_object_pos(self):
        return self.env.obj_init_pos

    def get_robot_bbox(self):
        eef_pos = self.env.get_endeff_pos()
        bbox = np.array([[eef_pos[0] - 0.068, eef_pos[1] - 0.065, eef_pos[2] - 0.06],
                         [eef_pos[0] + 0.068, eef_pos[1] + 0.065, eef_pos[2] + 10]])
        
        return bbox

    @staticmethod
    def pcd_clip(pcd, bbox):
        """
        pcd: (n, 6)
        bbox: (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        points_inside = np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1)
        points_outside = np.logical_not(points_inside)
        pcd_inside = pcd[points_inside]
        pcd_outside = pcd[points_outside]
        assert pcd_inside.shape[0] + pcd_outside.shape[0] == pcd.shape[0]
        return pcd_inside, pcd_outside

    @staticmethod
    def pcd_translate_inside_bbox(pcd, bbox, trans_vec):
        """
        Translate the points inside the bbox with trans_vec, leaving other points unchanged
        pcd: (n, 6)
        bbox: (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        trans_vec (3,)
        """
        pcd_inside, pcd_outside = MetaWorldEnv.pcd_clip(pcd, bbox)
        pcd_inside[:, :3] += trans_vec
        return np.concatenate([pcd_inside, pcd_outside], axis=0)
    
    @staticmethod
    def pcd_translate_outside_bbox(pcd, bbox, trans_vec):
        """
        Translate the points outside the bbox with trans_vec
        pcd: (n, 6)
        bbox: (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        trans_vec (3,)
        """
        pcd_inside, pcd_outside = MetaWorldEnv.pcd_clip(pcd, bbox)
        pcd_outside[:, :3] += trans_vec
        return np.concatenate([pcd_inside, pcd_outside], axis=0)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='assembly')
    args = parser.parse_args()
    env_name = args.env
    # env_name = args.env + '-v2'
    
    # env = MetaWorldEnv(env_name)
    # from metaworld.policies import SawyerAssemblyV2Policy
    # policy = SawyerAssemblyV2Policy()

    env = MetaWorldEnv("handle-press-side-spatial")
    from metaworld.policies import SawyerHandlePressSideSpatialV2Policy
    policy = SawyerHandlePressSideSpatialV2Policy()

    import pcd_visualizer
    vis = pcd_visualizer.pcd_visualizer()
    # def visualize_env(env, reset_mode):
    #     env.env.reset_mode = reset_mode
    #     obs = env.reset()
    #     rgb = env.get_rgb()
    #     plt.imsave(f"debug/{reset_mode}.png", rgb)

    #     pcd, _ = env.get_point_cloud()
    #     vis.save_visualization_to_file(pcd, f"debug/{reset_mode}.html")

    # reset_modes = ["low", "high", "single"]
    # for reset_mode in reset_modes:
    #     visualize_env(env, reset_mode)


    target_pos = env.get_target_pos()
    print("target_pos: ", target_pos)

    ee_pos = env.get_robot_state()[:3]
    print("ee_pos: ", ee_pos)

    pcd, _ = env.get_point_cloud()
    bbox_robot = env.get_robot_bbox()

    # trans_vec = np.array([0.1, 0.1, 0])

    trans_vec = np.array([0., 0., 0])

    # pcd_visualizer.visualize_pointcloud(pcd)
    pcd_visualizer.visualize_pointcloud(env.pcd_translate_outside_bbox(pcd, bbox_robot, trans_vec))


    # print(pcd[0].shape)

    

    for i in range(10):
        action = env.action_space.sample()
        # action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        env.get_point_cloud()
        main_img, wrist_img = env.get_rgb()
        plt.imsave("main_camera.png", main_img)
        plt.imsave("wrist_camera.png", wrist_img)
        # time.sleep(0.5)
        print(i, reward, done)
        if done:
            break
    env.close()
    cprint("MetaWorld env successfully closed", "green")

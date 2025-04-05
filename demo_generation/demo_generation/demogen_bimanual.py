"""
TO BE CLEANED
"""

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
import visualizer
import numpy as np
import copy
import os
import zarr
from termcolor import cprint
from pcd_proc import restore_and_filter_pcd
import imageio
from tqdm import tqdm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.spatial.transform import Rotation as R
import h5py
import matplotlib.pyplot as plt
import imageio
import os
from matplotlib.ticker import FormatStrFormatter

vfunc = np.vectorize("{:.3f}".format)
# sam_mask_root = "/home/dsy/shared/DemoGen-DP3/sam_mask"
PCD = True 
TASK_SETTING = {
    "banana": {
        "right_hand": {
            "stages": 2,
            "objects": {
                "object": "basket",
                "target": None,
            },
            "trans": [True, False],
            "rot": [False, False],
            # "stage_frame": [65, 270, 389], # [stage-1, pre-2, stage-2]
            "stage_frame": [22, 90, 130],
            "trans": [
                [[0.0, 0.0], [0.15, 0.0], [0.0, 0.15], [0.15, 0.15]],
            ],
        },
        "left_hand": {
            "stages": 2,
            "objects": {
                "object": "banana",
                "target": None,
            },
            "trans": [False, False],
            "rot": [True, False],
            # "stage_frame": [65, 389], # [1(rot), 2(rot again)]
            "stage_frame": [22, 130],
            "rot_z_degree": [
                [45, 0, 90, 135],
            ],
        }
    }
}

class DemoAugmenter:
    def __init__(self, input_zarr_path, interpolate_step_size=0.01):
        self._load_from_zarr(input_zarr_path)
        self.interpolate_step_size = interpolate_step_size
        assert(self.interpolate_step_size <= 0.01, "Interpolation step size should be less than 1 cm.")

    def _load_from_zarr(self, zarr_path):
        cprint(f"Loading data from {zarr_path}", "blue")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['left_state', 'left_action', 'right_state', 'right_action', 'point_cloud'])
        self.n_source_episodes = self.replay_buffer.n_episodes
        self.demo_name = zarr_path.split("/")[-1].split(".")[0]
        
    def get_bbox_from_pcd(self, pcd, margin=0.01):
            min_vals = np.min(pcd[:, :3], axis=0)
            max_vals = np.max(pcd[:, :3], axis=0)
            if isinstance(margin, float):
                margin = np.array([margin, margin, margin])
            elif isinstance(margin, (list, tuple)):
                margin = np.array(margin)
            min_vals -= margin
            max_vals += margin
            return np.array([min_vals, max_vals])
    
    # def get_objects_pcd(self, 
    #                     pcd, 
    #                     settings,
    #                     n_episodes, 
    #                     object_or_target="object", 
    #                     ):
    #     mask_path = f"{sam_mask_root}/{exp_name}/{settings["objects"][object_or_target]}_{n_episodes}.jpg"
    #     mask = imageio.imread(mask_path)
        
    #     threshold = 128
    #     mask = mask > threshold
    #     filtered_pcd = restore_and_filter_pcd(pcd, mask)
    #     # print(f"filtered_pcd shape: {filtered_pcd.shape}")
    #     # visualizer.visualize_pointcloud(filtered_pcd)
    #     return filtered_pcd

    def two_stage_pick_rot(self,
                            stage_frames,                  
                            states, 
                            actions,
                            point_clouds,
                            obj_bbox,
                            rot_bbox,
                            rot_center,
                            rot_z_degree,
                            rot_every_frame_degree=2.5,
                            ):
        ############## start of generating one episode ##############
        cur = 0
        gen_states = []
        gen_actions = []
        gen_pcds = []
        
        start_rot_frame = stage_frames[0]
        rot_again_frame = stage_frames[1]
        
        ############# stage {pre-1} starts #############
        for i in range(start_rot_frame):
            gen_actions.append(actions[cur])
            gen_states.append(states[cur])
            
            pcd_obj, pcd_ee = self.pcd_divide(point_clouds[cur], [obj_bbox])
            pcd_obj = self.pcd_rotate(pcd_obj, rot_center, rot_z_degree)
            gen_pcds.append(np.concatenate([pcd_ee, pcd_obj], axis=0))

            cur += 1
        ############## stage {pre-1} ends #############
        
        ############# hand rotation starts #############
        add_frame = int(rot_z_degree / rot_every_frame_degree)
        edit_action = actions[cur].copy()
        edit_state = states[cur].copy()
        edit_pcd = point_clouds[cur].copy()
        
        for i in range(add_frame):
            rot = R.from_euler('z', rot_every_frame_degree * (i+1), degrees=True)
            
            action_rot = R.from_euler('xyz', edit_action[3:6], degrees=False)
            action_rot = (action_rot * rot).as_euler('xyz', degrees=False)
            gen_actions.append(np.concatenate([edit_action[:3], action_rot, np.array([edit_action[6]])], axis=0))
            
            state_rot = R.from_euler('xyz', edit_state[3:6], degrees=False)
            state_rot = (state_rot * rot).as_euler('xyz', degrees=False)
            gen_states.append(np.concatenate([edit_state[:3], state_rot, np.array([edit_state[6]])], axis=0))
            
            pcd_obj, pcd_ee = self.pcd_divide(edit_pcd, [obj_bbox])
            pcd_ee_rot, pcd_ee_static = self.pcd_divide(pcd_ee, [rot_bbox])
            pcd_obj = self.pcd_rotate(pcd_obj, rot_center, rot_z_degree)
            pcd_ee_rot = self.pcd_rotate(pcd_ee_rot, rot_center, rot_every_frame_degree * (i+1))
            gen_pcds.append(np.concatenate([pcd_ee_rot, pcd_ee_static, pcd_obj], axis=0))
        
        ############# stage {1} starts #############
        is_stage_pre2 = cur >= rot_again_frame
        rot = R.from_euler('z', rot_z_degree, degrees=True)
        while not is_stage_pre2:
            # print(actions[cur])
            action_rot = R.from_euler('xyz', actions[cur][3:6], degrees=False)
            action_rot = (action_rot * rot).as_euler('xyz', degrees=False)
            gen_actions.append(np.concatenate([actions[cur][:3], action_rot, np.array([actions[cur][6]])], axis=0))
            
            state_rot = R.from_euler('xyz', states[cur][3:6], degrees=False)
            state_rot = (state_rot * rot).as_euler('xyz', degrees=False)
            gen_states.append(np.concatenate([states[cur][:3], state_rot, np.array([states[cur][6]])], axis=0))
            
            pcd_obj_ee = point_clouds[cur]
            pcd_obj_ee_rot, pcd_obj_ee_static = self.pcd_divide(pcd_obj_ee, [rot_bbox])
            pcd_obj_ee_rot = self.pcd_rotate(pcd_obj_ee_rot, rot_center, rot_z_degree)
            gen_pcds.append(np.concatenate([pcd_obj_ee_rot, pcd_obj_ee_static], axis=0))

            cur += 1
            is_stage_pre2 = cur >= rot_again_frame
        ############## stage {1} ends #############
        
        ############# hand rotation again #############
        add_frame = int(rot_z_degree / rot_every_frame_degree)
        edit_action = actions[cur].copy()
        edit_state = states[cur].copy()
        edit_pcd = point_clouds[cur].copy()
        
        for i in range(add_frame):
            rot = R.from_euler('z', rot_z_degree - rot_every_frame_degree * (i+1), degrees=True)
            
            action_rot = R.from_euler('xyz', edit_action[3:6], degrees=False)
            action_rot = (action_rot * rot).as_euler('xyz', degrees=False)
            gen_actions.append(np.concatenate([edit_action[:3], action_rot, np.array([edit_action[6]])], axis=0))
            
            state_rot = R.from_euler('xyz', edit_state[3:6], degrees=False)
            state_rot = (state_rot * rot).as_euler('xyz', degrees=False)
            gen_states.append(np.concatenate([edit_state[:3], state_rot, np.array([edit_state[6]])], axis=0))
            
            pcd_obj_ee = edit_pcd
            pcd_obj_ee_rot, pcd_obj_ee_static = self.pcd_divide(pcd_obj_ee, [rot_bbox])
            pcd_obj_ee_rot = self.pcd_rotate(pcd_obj_ee_rot, rot_center, rot_z_degree - rot_every_frame_degree * (i+1))
            gen_pcds.append(np.concatenate([pcd_obj_ee_rot, pcd_obj_ee_static], axis=0))
        
        return gen_states, gen_actions, gen_pcds
    
    def two_stage_pick_trans(self,
                            stage_frames,                  
                            states, 
                            actions,
                            point_clouds,
                            obj_bbox,
                            trans_vec_1,
                            ):
        ############## start of generating one episode ##############
        cur = 0
        gen_states = []
        gen_actions = []
        gen_pcds = []
        trans_vec_1 = step_trans_1 = np.concatenate([trans_vec_1, np.array([0])])
        stage_1_frame = stage_frames[0]
        pre_2_frame = stage_frames[1]
        stage_2_frame = stage_frames[2]
        
        ############# stage {pre-1} starts #############
        step_trans_1 = trans_vec_1 / (stage_1_frame-1)
        
        for i in range(stage_1_frame):
            gen_action = actions[cur][:3] + step_trans_1 * i
            gen_actions.append(np.concatenate([gen_action, actions[cur][3:]], axis=0))
            
            gen_state = states[cur][:3] + step_trans_1 * i
            gen_states.append(np.concatenate([gen_state, states[cur][3:]], axis=0))
            
            pcd_obj, pcd_ee = self.pcd_divide(point_clouds[cur], [obj_bbox])
            pcd_obj = self.pcd_translate(pcd_obj, trans_vec_1)
            pcd_ee = self.pcd_translate(pcd_ee, step_trans_1 * i)
            gen_pcds.append(np.concatenate([pcd_ee, pcd_obj], axis=0))

            cur += 1
        ############## stage {pre-1} ends #############
        
        ############# stage {1} starts #############
        is_stage_pre2 = cur >= pre_2_frame
        while not is_stage_pre2:
            gen_action = actions[cur][:3] + trans_vec_1
            gen_actions.append(np.concatenate([gen_action, actions[cur][3:]], axis=0))

            gen_state = states[cur][:3] + trans_vec_1
            gen_states.append(np.concatenate([gen_state, states[cur][3:]], axis=0))

            pcd_obj_ee = point_clouds[cur]
            pcd_obj_ee = self.pcd_translate(pcd_obj_ee, trans_vec_1)
            gen_pcds.append(pcd_obj_ee)

            cur += 1
            is_stage_pre2 = cur >= pre_2_frame
        ############## stage {1} ends #############

        ############# stage {pre2} starts #############
        step_trans_2 = trans_vec_1 / (stage_2_frame - pre_2_frame-1)

        for i in range(stage_2_frame - pre_2_frame):
            gen_action = actions[cur][:3] + trans_vec_1 - step_trans_2 * i
            gen_actions.append(np.concatenate([gen_action, actions[cur][3:]], axis=0))
            
            gen_state = states[cur][:3] + trans_vec_1 - step_trans_2 * i
            gen_states.append(np.concatenate([gen_state, states[cur][3:]], axis=0))

            pcd_obj_ee = point_clouds[cur]
            pcd_obj_ee = self.pcd_translate(pcd_obj_ee, trans_vec_1 - step_trans_2 * i)
            gen_pcds.append(pcd_obj_ee)

            cur += 1
        ############## stage {pre2} ends #############   
        
        return gen_states, gen_actions, gen_pcds
    
    def banana_augment(self, exp_name, env_name="banana", aug_name="4x4", downsample=1024):      
        generated_episodes = []
        right_settings = TASK_SETTING[env_name]["right_hand"]
        left_settings = TASK_SETTING[env_name]["left_hand"]            
        for i in tqdm(range(self.n_source_episodes)):
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            # visualizer.visualize_pointcloud(pcds[0])
            
            left_frames = left_settings["stage_frame"]
            right_frames = right_settings["stage_frame"]
            # pcd_left_obj = self.get_objects_pcd(pcds[0], left_settings, i, "object")
            # pcd_right_obj = self.get_objects_pcd(pcds[0], right_settings, i, "object")   
            # right_obj_bbox = self.get_bbox_from_pcd(pcd_right_obj)
            # left_obj_bbox = self.get_bbox_from_pcd(pcd_left_obj)
            
            right_obj_bbox = np.array([[0.5, -0.42, 0.64], [0.8, -0.18, 0.75]])
            left_obj_bbox =np.array([[0.63, 0.1, 0.64], [0.73, 0.2, 0.72]])
            
            rot_bbox = np.array([[0.0, 0.05, 0.0], [1, 1, 0.88]])
            rot_center = np.array([0.616, 0.154, 0.0])
            
            right_trans_vecs = right_settings["trans"][0]
            left_rot_z_degrees = left_settings["rot_z_degree"][0]
            
            left_pcds = []
            right_pcds = []
            divide_bbox = np.array([[-1e6, 0.05, -1e6], [1e6, 1e6, 1e6]])
            for pcd in pcds:
                masks = []
                masks.append(np.all(pcd[:, :3] > divide_bbox[0], axis=1) & np.all(pcd[:, :3] < divide_bbox[1], axis=1))
                masks.append(np.logical_not(np.any(masks, axis=0)))
                left_pcds.append(pcd[masks[0]])
                right_pcds.append(pcd[masks[1]])

            # for index in range(len(left_pcds)):
            #     left_pcd = left_pcds[index]
            #     right_pcd = right_pcds[index]
            #     print(left_pcd.shape, right_pcd.shape)
            #     assert left_pcd.shape[0] + right_pcd.shape[0] == pcds[index].shape[0]
            
            for right_trans_vec in tqdm(right_trans_vecs):
                for left_rot_z_degree in tqdm(left_rot_z_degrees):
                    left_gen_states, left_gen_actions, left_gen_pcds = self.two_stage_pick_rot(
                        stage_frames=left_frames,                  
                        states=source_demo["left_state"].copy(),
                        actions=source_demo["left_action"].copy(),
                        point_clouds=left_pcds,
                        obj_bbox=left_obj_bbox,
                        rot_bbox=rot_bbox,
                        rot_center=rot_center,
                        rot_z_degree=left_rot_z_degree,
                        rot_every_frame_degree=2.5)
                    
                    # for index in range(len(left_pcds)):
                    #     left_gen_pcd = left_gen_pcds[index]
                    #     left_pcd = left_pcds[index]
                        
                    #     if not left_gen_pcd.shape[0] == left_pcd.shape[0]:
                    #         print(f"left_gen_pcd shape: {left_gen_pcd.shape}, left_pcd shape: {left_pcd.shape}")
                    
                    right_gen_states, right_gen_actions, right_gen_pcds = self.two_stage_pick_trans(
                        stage_frames=right_frames,      
                        states=source_demo["right_state"].copy(),
                        actions=source_demo["right_action"].copy(),
                        point_clouds=right_pcds,
                        obj_bbox=right_obj_bbox,
                        trans_vec_1=right_trans_vec,
                    )
                    
                    # for index in range(len(right_pcds)):
                    #     right_gen_pcd = right_gen_pcds[index]
                    #     right_pcd = right_pcds[index]
                    #     assert right_gen_pcd.shape[0] == right_pcd.shape[0]
                    
                    left_frames_num = len(left_gen_states)
                    right_frames_num = len(right_gen_states)
                    print(f"left_frames_num: {left_frames_num}, right_frames_num: {right_frames_num}")
                    
                    if left_frames_num > right_frames_num:
                        padding_frames = left_frames_num - right_frames_num
                        right_gen_states += [right_gen_states[-1]] * padding_frames
                        right_gen_actions += [right_gen_actions[-1]] * padding_frames
                        right_gen_pcds += [right_gen_pcds[-1]] * padding_frames
                        print(f"After padding - right_frames_num: {len(right_gen_states)}")

                    ######## start to merge left and right hand data ########
                    left_gen_states = np.array(left_gen_states)
                    left_gen_actions = np.array(left_gen_actions)
                    right_gen_states = np.array(right_gen_states)
                    right_gen_actions = np.array(right_gen_actions)
                    
                    gen_pcds = []
                    for gen_frame_num in range(len(left_gen_states)):
                        left_pcd = left_gen_pcds[gen_frame_num]
                        right_pcd = right_gen_pcds[gen_frame_num]
                        pcd = np.concatenate([left_pcd, right_pcd], axis=0)
                        assert pcd.shape[0] > downsample
                        idx = np.random.choice(pcd.shape[0], 800, replace=False)
                        pcd = pcd[idx]
                        gen_pcds.append(pcd)
                    
                    stage_2_frame = right_frames[2]
                    later_pcds = source_demo["point_cloud"][stage_2_frame:]
                    for pcd in later_pcds:
                        assert pcd.shape[0] > downsample
                        idx = np.random.choice(pcd.shape[0], 800, replace=False)
                        pcd = pcd[idx]
                        gen_pcds.append(pcd)
                    
                    generated_episode = {
                        "left_state": np.concatenate([left_gen_states, source_demo["left_state"][stage_2_frame:]], axis=0),
                        "left_action": np.concatenate([left_gen_actions, source_demo["left_action"][stage_2_frame:]], axis=0),
                        "right_state": np.concatenate([right_gen_states, source_demo["right_state"][stage_2_frame:]], axis=0),
                        "right_action": np.concatenate([right_gen_actions, source_demo["right_action"][stage_2_frame:]], axis=0),
                        "point_cloud": gen_pcds,
                    }
                    generated_episodes.append(generated_episode)

                    if PCD:
                        print(f"Saving video for episode {i}")
                        self.point_cloud_to_video_auto(generated_episode["point_cloud"], f'/home/dsy/shared/DemoGen-DP3/video_new/{exp_name}_{aug_name}/{i}_{left_rot_z_degree}_{right_trans_vec}-1.mp4', elev=80, azim=180)
                        self.point_cloud_to_video_auto(generated_episode["point_cloud"], f'/home/dsy/shared/DemoGen-DP3/video_new/{exp_name}_{aug_name}/{i}_{left_rot_z_degree}_{right_trans_vec}-2.mp4', elev=30, azim=45)
                    ############## end of generating one episode ##############
        
        save_episode_dir = f"/home/dsy/shared/DemoGen-DP3/demo/{exp_name}/{exp_name}_{aug_name}.zarr"
        self.save_episodes(generated_episodes, save_episode_dir) 
        
        
    @staticmethod
    def point_cloud_to_video_auto(point_clouds, output_file, fps=15, elev=30, azim=45):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        all_points = np.concatenate(point_clouds, axis=0)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = imageio.get_writer(output_file, fps=fps)

        for frame, points in enumerate(point_clouds):
            ax.clear()
            color = points[:, 3:] / 255
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')

            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            ax.set_zlim(min_vals[2], max_vals[2])

            ax.view_init(elev=elev, azim=azim)
            ax.text2D(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=14, 
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)

        writer.close()
        plt.close(fig)
    
    @staticmethod
    def point_cloud_to_video(point_clouds, output_file, fps=15, elev=30, azim=45):
        """
        Converts a sequence of point cloud frames into a video.

        Args:
            point_clouds (list): A list of (N, 6) numpy arrays representing the point clouds.
            output_file (str): The path to the output video file.
            fps (int, optional): The frames per second of the output video. Defaults to 15.
            elev (float, optional): The elevation angle (in degrees) for the 3D view. Defaults to 30.
            azim (float, optional): The azimuth angle (in degrees) for the 3D view. Defaults to 45.
        """
        fig = plt.figure(figsize=(8, 6), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        all_points = np.concatenate(point_clouds, axis=0)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        writer = imageio.get_writer(output_file, fps=fps)

        for points in point_clouds:
            ax.clear()
            color = points[:, 3:] / 255
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')

            ax.set_box_aspect([1.6, 1.6, 1])
            ax.set_xlim(0.1, 0.8)
            ax.set_ylim(-0.3, 0.4)
            ax.set_zlim(0.1, 0.4)

            x_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            y_ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
            z_ticks = [0.1, 0.2, 0.3, 0.4]
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.zaxis.set_major_locator(FixedLocator(z_ticks))
            
            formatter = FormatStrFormatter('%.1f')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.zaxis.set_major_formatter(formatter)

            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)

        writer.close()
        plt.close(fig)

    def save_episodes(self, generated_episodes, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cprint(f"Saving data to {save_dir}", "green")

        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        left_state_arrays = np.concatenate([ep["left_state"] for ep in generated_episodes], axis=0)
        right_state_arrays = np.concatenate([ep["right_state"] for ep in generated_episodes], axis=0)
        point_cloud_arrays = np.concatenate([ep["point_cloud"] for ep in generated_episodes], axis=0)
        left_action_arrays = np.concatenate([ep["left_action"] for ep in generated_episodes], axis=0)
        right_action_arrays = np.concatenate([ep["right_action"] for ep in generated_episodes], axis=0)
        episode_ends = []
        count = 0
        for ep in generated_episodes:
            count += len(ep["left_state"])
            episode_ends.append(count)
        episode_ends_arrays = np.array(episode_ends)
        print(episode_ends_arrays)

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        state_chunk_size = (100, left_state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        action_chunk_size = (100, left_action_arrays.shape[1])
        
        zarr_data.create_dataset('left_state', data=left_state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('right_state', data=right_state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('right_action', data=right_action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('left_action', data=left_action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

        cprint(f'-'*50, 'cyan')
        cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
        cprint(f'left_state shape: {left_state_arrays.shape}, range: [{np.min(left_state_arrays)}, {np.max(left_state_arrays)}]', 'green')
        cprint(f'right_state shape: {right_state_arrays.shape}, range: [{np.min(right_state_arrays)}, {np.max(right_state_arrays)}]', 'green')
        cprint(f'left_action shape: {left_action_arrays.shape}, range: [{np.min(left_action_arrays)}, {np.max(left_action_arrays)}]', 'green')
        cprint(f'right_action shape: {right_action_arrays.shape}, range: [{np.min(right_action_arrays)}, {np.max(right_action_arrays)}]', 'green')
        cprint(f'Saved zarr file to {save_dir}', 'green')

        # save to hdf5
        save_dir = save_dir.replace('.zarr', '.hdf5')
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('point_cloud', data=point_cloud_arrays, compression='gzip')
            f.create_dataset('episode_ends', data=episode_ends_arrays, compression='gzip')
            f.create_dataset('left_state', data=left_state_arrays, compression='gzip')
            f.create_dataset('right_state', data=right_state_arrays, compression='gzip')
            f.create_dataset('left_action', data=left_action_arrays, compression='gzip')
            f.create_dataset('right_action', data=right_action_arrays, compression='gzip')
        cprint(f'Saved hdf5 file to {save_dir}', 'green')

    @staticmethod
    def pcd_divide(pcd, bbox_list):
        """
        pcd: (n, 6)
        bbox_list: list of (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        :return: list of pcds
        """
        masks = []
        selected_pcds = []
        for bbox in bbox_list:
            assert np.array(bbox).shape == (2, 3)
            masks.append(np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1))
        # add the rest of the points to the last mask
        masks.append(np.logical_not(np.any(masks, axis=0)))
        for mask in masks:
            selected_pcds.append(pcd[mask])
        # assert np.sum([len(p) for p in selected_pcds]) == pcd.shape[0]
        return selected_pcds

    @staticmethod
    def pcd_rotate(pcd, rot_center, rot_z_degree):
        rot_matrix = R.from_euler('z', rot_z_degree, degrees=True)
        points_xyz = pcd[:, :3]
        translated_points = points_xyz - rot_center
        rotated_points = rot_matrix.apply(translated_points)
        final_points_xyz = rotated_points + rot_center
        final_points = np.concatenate([final_points_xyz, pcd[:, 3:]], axis=1)      
        return final_points
          
    @staticmethod
    def pcd_translate(pcd, trans_vec):
        """
        Translate the points with trans_vec
        pcd: (n, 6)
        trans_vec (3,)
        """
        pcd_ = pcd.copy()
        pcd_[:, :3] += trans_vec
        return pcd_

if __name__ == '__main__':
    exp_name = "banana_3x"
    zarr_path = f"/home/dsy/shared/DemoGen-DP3/demo/{exp_name}.zarr"
    augmenter = DemoAugmenter(zarr_path)
    gen_demos = augmenter.banana_augment(exp_name, env_name="banana", aug_name="4x4")     
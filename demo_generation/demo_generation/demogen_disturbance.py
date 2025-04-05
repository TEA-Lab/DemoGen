"""
TO BE CLEANED
"""

import itertools
from re import T
from turtle import st
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from gym.envs.unittest import cube_crash
from regex import I
import visualizer
import numpy as np
import os
import zarr
from termcolor import cprint
from pcd_proc import restore_and_filter_pcd
import imageio
from tqdm import tqdm
from matplotlib.ticker import FixedLocator, FormatStrFormatter

vfunc = np.vectorize("{:.3f}".format)

MOVING_NUM = 3
PCD = True 
SMALL_RANGE_SHIFT = True

STAGE_FRAME = {
    "sauce": { # 90
        2: {
            "stage_steps": 5,
            "stage-0": 0,
            "stage-1": 10,
            "stage-2": 40,
        },
        3: {
            "stage_steps": 5,
            "stage-0": 0,
            "stage-1": 10,
            "stage-2": 28,
            "stage-3": 70, 
        },
        4: {
            "stage_steps": 5,
            "stage-0": 0,
            "stage-1": 10,
            "stage-2": 30,
            "stage-3": 50, 
            "stage-4": 70, 
        },
    },
}

TRANS_RANGE = {
    "sauce": {
        "move" : {
            "steps": [[0.075, 0.0, 0.0], [0.0, 0.075, 0.0], [-0.075, 0.0, 0.0], [0.0, -0.075, 0.0],
                      [0.075, 0.075, 0.0], [-0.075, 0.075, 0.0], [-0.075, -0.075, 0.0], [0.075, -0.075, 0.0]],
            "origin_offsets": [[0.0,  0.0,   0.0, 0.15],
                               [0.0, 0.15, -0.15,  0.0]],
        },
        "move-single" : {
            "range":[[-0.015, -0.015], [0.015, 0.015]],
            "steps": [[0.075, 0.0, 0.0], [0.0, 0.075, 0.0], [-0.075, 0.0, 0.0], [0.0, -0.075, 0.0],
                      [0.075, 0.075, 0.0], [-0.075, 0.075, 0.0], [-0.075, -0.075, 0.0], [0.075, -0.075, 0.0]],
            "origin_offsets": [[0.0],
                               [0.0]],
        },
        "5points" : {
            "range":[[-0.015, -0.015], [0.015, 0.015]],
            "groups": [
                        [[0.0, 0.0, 0.0], [0.075, 0.0, 0.0], [0.075, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.075, 0.0], [0.0, 0.075, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, -0.075, 0.0], [0.0, -0.075, 0.0]],
                        [[0.0, 0.0, 0.0], [0.075, 0.075, 0.0], [0.075, 0.075, 0.0]],
                        [[0.0, 0.0, 0.0], [0.075, -0.075, 0.0], [0.075, -0.075, 0.0]],
                       ]
        },
        "test" : {
            "range":[[-0.015, -0.015], [0.015, 0.015]],
            "groups": [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                       ]
        }
    },
}

OBJECTS = {
    "sauce": {
        "object": "plate",
    }
}

def is_in_workspace(x, y):
    return 0.0 <= x <= 0.15 and -0.3 <= y <= 0.3 and y <= x + 0.3 and y <= -x + 0.3

class DemoAugmenter:
    def __init__(self, input_zarr_path, interpolate_step_size=0.01):
        self._load_from_zarr(input_zarr_path)
        self.interpolate_step_size = interpolate_step_size
        assert(self.interpolate_step_size <= 0.01, "Interpolation step size should be less than 1 cm.")

    def _load_from_zarr(self, zarr_path):
        cprint(f"Loading data from {zarr_path}", "blue")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])
        self.n_source_episodes = self.replay_buffer.n_episodes
        self.demo_name = zarr_path.split("/")[-1].split(".")[0]
    
    def generate_move_trans_groups(self, origin_offsets, steps):
        trans_groups = []
        for x_offset, y_offset in zip(origin_offsets[0], origin_offsets[1]):
            for step_sequence in itertools.product(steps, repeat=MOVING_NUM):
                trans_group = []
                trans_group.append([x_offset, y_offset, 0.0])
                current_x, current_y = x_offset, y_offset
                valid_sequence = True
                
                for step in step_sequence:
                    next_x = current_x + step[0]
                    next_y = current_y + step[1]
                    if not is_in_workspace(next_x, next_y):
                        valid_sequence = False
                        break
                    trans_group.append(step)
                    current_x, current_y = next_x, next_y
                
                if valid_sequence:
                    trans_groups.append(trans_group)

        print(f"Generated {len(trans_groups)} valid move sequences")
        return trans_groups
    
    def small_range_shift(self, trans_groups, trans_range, shift_num):
        n_side = int(np.sqrt(shift_num))
        if n_side ** 2 != shift_num:
            raise ValueError("shift_num must be a squared number")
        x_min, x_max, y_min, y_max = trans_range[0][0], trans_range[1][0], trans_range[0][1], trans_range[1][1]
        x_values = [x_min + i / (n_side - 1) * (x_max - x_min) for i in range(n_side)]
        y_values = [y_min + i / (n_side - 1) * (y_max - y_min) for i in range(n_side)]
        shift_list = list(set([(x, y, 0) for x in x_values for y in y_values]))
        
        trans_groups_shifted = []
        for trans_group in trans_groups:
            # Generate all possible combinations of shifts for this trans_group
            shift_combinations = list(itertools.product(shift_list, repeat=len(trans_group)))
            
            for shift_combo in shift_combinations:
                trans_group_shifted = [
                    [x + y for x, y in zip(trans_vec, shift)]
                    for trans_vec, shift in zip(trans_group, shift_combo)
                ]
                trans_groups_shifted.append(trans_group_shifted)
        
        print(f"Generated {len(trans_groups_shifted)} shifted translation groups")
        # print(f"{trans_groups_shifted}")
        return trans_groups_shifted
                
    def one_move_greedy_fuse_augment(self, exp_name, env_name, aug_setting, shift_num=9):
        def get_objects_pcd(pcd, exp_name, env_name, n_episodes, object_or_target="object"):
            mask = imageio.imread(f"/home/dsy/shared/DemoGen-DP3/sam_mask/{exp_name}/{OBJECTS[env_name][object_or_target]}_{n_episodes}.jpg")
            threshold = 128 # 128
            # if mask.shape[2] == 3:
            #     mask = mask[:, :, 0] * 255
            mask = mask > threshold
            filtered_pcd = restore_and_filter_pcd(pcd, mask)
            return filtered_pcd
        
        def get_bbox_from_pcd(pcd):
                min_vals = np.min(pcd[:, :3], axis=0)
                max_vals = np.max(pcd[:, :3], axis=0)
                return np.array([min_vals, max_vals])

        if "groups" in TRANS_RANGE[env_name][aug_setting]:
            object_trans_groups = TRANS_RANGE[env_name][aug_setting]["groups"]
            assert len(object_trans_groups[0]) == MOVING_NUM + 1
        else:
            steps = TRANS_RANGE[env_name][aug_setting]["steps"]
            origin_offsets = TRANS_RANGE[env_name][aug_setting]["origin_offsets"]
            object_trans_groups = self.generate_move_trans_groups(origin_offsets, steps)

        if SMALL_RANGE_SHIFT:
            assert "range" in TRANS_RANGE[env_name][aug_setting]
            trans_range = TRANS_RANGE[env_name][aug_setting]["range"]
            object_trans_groups = self.small_range_shift(object_trans_groups, trans_range, shift_num)
        # object_trans_groups = np.array([[[0.0, 0.0, 0.0], [0.075, 0.0, 0.0],
        #               [0.0, -0.075, 0.0], [-0.075, 0.0, 0.0]]])
        generated_episodes = []
        frames = STAGE_FRAME[env_name][MOVING_NUM]
        stage_steps = frames["stage_steps"]

        for i in tqdm(range(self.n_source_episodes)):
            source_demo = self.replay_buffer.get_episode(i)
            pcds = source_demo["point_cloud"]
            pcd_obj = get_objects_pcd(pcds[0], exp_name, env_name, i, "object")       
            obj_bbox = get_bbox_from_pcd(pcd_obj)
            obj_bbox[0] -= 0.01
            obj_bbox[1] += 0.01
            obj_bbox[0][2] = 0.0
            obj_bbox[1][2] -= 0.01
            print(f"obj_bbox: {obj_bbox}")

            for object_trans_group in tqdm(object_trans_groups):
                # print(f"using translation groups: ", object_trans_group)
                current_stage_index = 0
                current_frame = 0
                traj_states = []
                traj_actions = []
                traj_pcds = []
                trans_sofar = np.zeros(3)
                    
                for obj_trans_vec in object_trans_group:
                    current_stage = f"stage-{current_stage_index}"
                    stage_start_frame = frames[current_stage]
                    stage_end_frame = stage_start_frame + stage_steps
                    
                    trans_togo = obj_trans_vec.copy()
                    start_pos = (source_demo["state"][stage_start_frame][:3] - 
                                source_demo["action"][stage_start_frame][:3])
                    end_pos = source_demo["state"][stage_end_frame][:3] + trans_togo
                    step_action = (end_pos - start_pos) / stage_steps
                    
                    for _ in range(stage_steps):
                        # print(f"moving!! stage-{current_stage_index} current_frame: {current_frame}")
                        source_action = source_demo["action"][current_frame]
                        traj_actions.append(np.concatenate([step_action, source_action[3:]], axis=0))
                        trans_this_frame = step_action - source_action[:3]

                        # "state" and "point_cloud" consider the accumulated translation
                        state = source_demo["state"][current_frame].copy()
                        state[:3] += (trans_this_frame + trans_sofar)
                        traj_states.append(state)
                        
                        source_pcd = source_demo["point_cloud"][current_frame].copy()
                        pcd_obj, pcd_robot = self.pcd_divide(source_pcd, [obj_bbox])
                        pcd_obj = self.pcd_translate(pcd_obj, obj_trans_vec + trans_sofar)
                        pcd_robot = self.pcd_translate(pcd_robot, trans_this_frame + trans_sofar)
                        traj_pcds.append(np.concatenate([pcd_robot, pcd_obj], axis=0))

                        current_frame += 1
                    
                    trans_sofar += trans_togo
                    # print(f"trans_sofar: {trans_sofar}")
                    end = frames[f"stage-{current_stage_index + 1}"] if current_stage_index < MOVING_NUM else len(source_demo["state"])
                    while current_frame < end:          
                        # print(f"stage-{current_stage_index} current_frame: {current_frame}")           
                        action = source_demo["action"][current_frame].copy()
                        traj_actions.append(action)

                        state = source_demo["state"][current_frame].copy()
                        state[:3] += trans_sofar
                        traj_states.append(state)
                        
                        pcd_obj_robot = source_demo["point_cloud"][current_frame].copy()
                        pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans_sofar)
                        traj_pcds.append(pcd_obj_robot)

                        current_frame += 1
                    
                    current_stage_index += 1

                generated_episode = {
                    "state": traj_states,
                    "action": traj_actions,
                    "point_cloud": traj_pcds
                }
                generated_episodes.append(generated_episode)

                if PCD:
                    print(f"Saving video for episode {i}")
                    self.point_cloud_to_video(generated_episode["point_cloud"], f'/home/dsy/shared/DemoGen-DP3/video_new/{exp_name}-{aug_setting}-{MOVING_NUM}/{i}_{obj_trans_vec}.mp4', elev=20, azim=30)
                ############## end of generating one episode ##############
        
        # save the generated episodes
        if SMALL_RANGE_SHIFT:
            aug_setting = f"{aug_setting}-shift{shift_num}"
        save_episode_dir = f"/home/dsy/shared/DemoGen-DP3/demo/{exp_name}/{exp_name}_{aug_setting}_{MOVING_NUM}.zarr"
        self.save_episodes(generated_episodes, save_episode_dir)
        
    @staticmethod
    def point_cloud_to_video(point_clouds, output_file, fps=15, elev=30, azim=45):
        import matplotlib.pyplot as plt
        import imageio
        import os
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

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

        for frame, points in enumerate(point_clouds):
            ax.clear()
            color = points[:, 3:] / 255
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')

            ax.set_box_aspect([1.6, 1.6, 1])
            # ax.set_aspect('equal')
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
            # ax.text2D(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=14, 
            #       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)

        writer.close()
        plt.close(fig)

    # @staticmethod
    # def point_cloud_to_video(point_clouds, output_file, fps=5, elev=30, azim=45):
    #     import matplotlib.pyplot as plt
    #     import imageio
    #     """
    #     point_clouds: (N, 6) numpy
    #     elev: 仰角（度）
    #     azim: 方位角（度）
    #     """
        
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     all_points = np.concatenate(point_clouds, axis=0)
    #     min_vals = np.min(all_points, axis=0)
    #     max_vals = np.max(all_points, axis=0)
        
    #     os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     writer = imageio.get_writer(output_file, fps=fps)
        
    #     for frame, points in enumerate(point_clouds):
    #         ax.clear()
    #         color = points[:, 3:] / 255
    #         ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='.')
            
    #         ax.set_xlim(min_vals[0], max_vals[0])
    #         ax.set_ylim(min_vals[1], max_vals[1])
    #         ax.set_zlim(min_vals[2], max_vals[2])
            
    #         ax.view_init(elev=elev, azim=azim)
            
    #         ax.text2D(0.05, 0.95, f'Frame: {frame}', transform=ax.transAxes, fontsize=14, 
    #               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
    #         fig.canvas.draw()
    #         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
    #         writer.append_data(img)
        
    #     writer.close()
    #     plt.close(fig)
        

    def _examine_episode(self, episode, aug_setting, episode_id, obj_trans, tar_trans):
        """
        Examine the episode to see if the point cloud is correct
        """
        cprint(f"Examine episode {episode_id}", "green")
        
        debug_save_dir = f"/home/zhengrong/data/dp3/debug/demo_generation/{self.demo_name}/{aug_setting}/{episode_id}-{vfunc(obj_trans)}-{vfunc(tar_trans)}"
        os.makedirs(debug_save_dir, exist_ok=True)
        cprint(f"Saving episode examination to {debug_save_dir}", "yellow")
        vis = visualizer.Visualizer()
        ep_len = episode["state"].shape[0]
        pcds = []
        for i in range(0, 200, 1):
            # print(f"Frame {i}")
            cprint(f"action {i}: {vfunc(episode['action'][i])}", "blue")
            cprint(f"state {i}: {vfunc(episode['state'][i][:3])}", "blue")
            pcd = episode["point_cloud"][i]
            pcds.append(pcd)
            vis.save_visualization_to_file(pointcloud=pcd, file_path=os.path.join(debug_save_dir, f"frame_{i}.html"))

        # vis.preview_in_open3d(pcds)

    def _examine_actions(self, demo_trajectory):
        vfunc = np.vectorize("{:.2e}".format)
        actions = demo_trajectory["action"][:30]
        ee_actions = actions[:, :3]
        intensity = np.linalg.norm(ee_actions, axis=1)
        print(f"ee_actions: {vfunc(ee_actions)}")
        print(f"inensity: {vfunc(intensity)}")

    def save_episodes(self, generated_episodes, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cprint(f"Saving data to {save_dir}", "green")
        # self.replay_buffer.save_to_store(save_dir)

        zarr_root = zarr.group(save_dir)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        state_arrays = np.concatenate([ep["state"] for ep in generated_episodes], axis=0)
        point_cloud_arrays = np.concatenate([ep["point_cloud"] for ep in generated_episodes], axis=0)
        action_arrays = np.concatenate([ep["action"] for ep in generated_episodes], axis=0)
        episode_ends = []
        count = 0
        for ep in generated_episodes:
            count += len(ep["state"])
            episode_ends.append(count)
        episode_ends_arrays = np.array(episode_ends)
        print(episode_ends_arrays)

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
        action_chunk_size = (100, action_arrays.shape[1])
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

        cprint(f'-'*50, 'cyan')
        # print shape
        cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
        cprint(f'Saved zarr file to {save_dir}', 'green')

        # save to hdf5
        # import h5py
        # save_dir = save_dir.replace('.zarr', '.hdf5')
        # with h5py.File(save_dir, 'w') as f:
        #     f.create_dataset('state', data=state_arrays, compression='gzip')
        #     f.create_dataset('point_cloud', data=point_cloud_arrays, compression='gzip')
        #     f.create_dataset('action', data=action_arrays, compression='gzip')
        #     f.create_dataset('episode_ends', data=episode_ends_arrays, compression='gzip')
        # cprint(f'Saved hdf5 file to {save_dir}', 'green')

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
        assert np.sum([len(p) for p in selected_pcds]) == pcd.shape[0]
        return selected_pcds

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

    @staticmethod
    def pcd_select_and_translate(pcd, bbox, trans_vec):
        """
        pcd: (n, 6)
        bbox: (2, 3), [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        trans_vec: (3,)
        """
        mask = np.all(pcd[:, :3] > bbox[0], axis=1) & np.all(pcd[:, :3] < bbox[1], axis=1)
        return DemoAugmenter.pcd_translate(pcd[mask], trans_vec)
    
    @staticmethod
    def translate_all_frames(source_demo, trans_vec, start_frame=0):
        """
        Translate all frames in the source demo by trans_vec from start_frame
        """
        source_states = source_demo["state"]
        source_actions = source_demo["action"]
        source_pcds = source_demo["point_cloud"]
        # states = source_states[start_frame:] + np.array([*trans_vec, *trans_vec, *trans_vec])
        states = source_states[start_frame:].copy()
        states[:, :3] += trans_vec
        actions = source_actions[start_frame:]
        pcds = source_pcds[start_frame:].copy()   # [T, N_points, 6]
        pcds[:, :, :3] += trans_vec
        assert states.shape[0] == actions.shape[0] == pcds.shape[0] == source_states.shape[0] - start_frame
        return {
            "state": states,
            "action": actions,
            "point_cloud": pcds
        }

if __name__ == '__main__':
    exp_name = "sauce-new"
    zarr_path = f"/home/dsy/shared/DemoGen-DP3/demo/{exp_name}.zarr"
    augmenter = DemoAugmenter(zarr_path)

    env_name = "sauce"
    gen_setting = "move-single" 
    shift_num = 9
    gen_demos = augmenter.one_move_greedy_fuse_augment(exp_name, env_name, gen_setting, shift_num)

    
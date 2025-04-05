"""
TO BE CLEANED
"""

from diffusion_policies.common.replay_buffer import ReplayBuffer
import pcd_visualizer
import numpy as np
import os
import zarr
from termcolor import cprint
from tqdm import tqdm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import fpsample
import trimesh

from demo_generation.demogen import DemoGen


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

def pcd_translate(pcd, trans_vec):
    """
    Translate the points with trans_vec
    pcd: (n, 6)
    trans_vec (3,)
    """
    pcd_ = pcd.copy()
    pcd_[:, :3] += trans_vec
    return pcd_

def distracted_pcds(mesh_root, mesh_list, points_num=20000):
    def y_up_to_z_up(pcd):
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        pcd_transformed = pcd.copy()
        pcd_transformed[:, :3] = pcd[:, :3] @ rotation_matrix.T
        return pcd_transformed

    distracted_pcd_list = []
    for mesh_name in mesh_list:
        if mesh_name == "ball.obj":
            crop_bbox = [[0, -0.066, 0.01], [1, 0.066, 0.20]]
            trans_vec = np.array([0.422, 0.062, 0.02])
        elif mesh_name == "cone.obj":
            crop_bbox = [[0, -1, 0.01], [1, 1, 0.15]]
            trans_vec = np.array([0.414, 0.053, 0.02])
        elif mesh_name == "bottle.obj":
            crop_bbox = [[0, -1, 0.01], [1, 1, 0.20]]
            trans_vec = np.array([0.438, 0.057, 0.02])

        mesh_path = os.path.join(mesh_root, mesh_name)
        mesh = trimesh.load_mesh(mesh_path)
        pcd = mesh.sample(points_num)
        pcd = np.concatenate([pcd, np.zeros((pcd.shape[0], 3))], axis=1)
        pcd = y_up_to_z_up(pcd)

        pcd_cropped, _ = pcd_divide(pcd, [crop_bbox])
        pcd_cropped_array = np.vstack(pcd_cropped) 
        pcd_trans = pcd_translate(pcd_cropped_array, trans_vec)

        points_xyz = pcd_trans[..., :3]
        sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, 1024, h=3)
        pcd_trans = pcd_trans[sample_indices]
        
        # pcd_visualizer.visualize_pointcloud(pcd_trans)

        distracted_pcd_list.append(pcd_trans)

    return distracted_pcd_list

def load_distracted_pcds(distract_pcd_root):
    distracted_pcd_list = []
    for file in os.listdir(distract_pcd_root):
        if file.endswith(".npy"):
            pcd = np.load(os.path.join(distract_pcd_root, file))
            distracted_pcd_list.append(pcd)

    return distracted_pcd_list

def load_crop_obj(distract_pcd_root):
    distracted_pcd_list = []
    for file in os.listdir(distract_pcd_root):
        if file.endswith(".npy"):
            pcd = np.load(os.path.join(distract_pcd_root, file))
            pcd_cropped, _ = pcd_divide(pcd, [[[-10, -0.1, -0.01], [10, 0.155, 0.18]]])
            distracted_pcd_list.append(pcd_cropped)
            pcd_visualizer.visualize_pointcloud(pcd_cropped)

    return distracted_pcd_list

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

    def crop_distracted_obj(self):
        for i in range(self.n_source_episodes):
            source_demo = self.replay_buffer.get_episode(i)
            pcd = source_demo["point_cloud"][0]
            pcd_cropped, _ = self.pcd_divide(pcd, [[[-10, -0.1, -0.01], [10, 0.155, 0.18]]])
            # pcd_visualizer.visualize_pointcloud(pcd_cropped)
            # save to npy
            os.makedirs(f"/home/zhengrong/projects/Spatial-Generalization/data/demo/{self.demo_name}", exist_ok=True)
            np.save(f"/home/zhengrong/projects/Spatial-Generalization/data/demo/{self.demo_name}/{i}_cropped.npy", pcd_cropped)

    def distracted_object_augment(self, exp_name, distracted_pcd_list, video_pcd=False): 
        generated_episodes = []

        for i in tqdm(range(self.n_source_episodes)):
            source_demo = self.replay_buffer.get_episode(i)
            
            pre_2_frame = 38
            stage_2_frame = 49
            num_points = 1024
            tar_bbox = [[-10, -0.42, -0.01], [10, -0.1, 0.08]]
            crop_bbox = [[-10, -10, -0.01], [10, 10, 0.50]]
            interpolate_steps = 7
            interpolate_step_size = 0.02

            for index, distract_pcd in tqdm(enumerate(distracted_pcd_list)):
                ############## start of generating one episode ##############
                current_frame = 0

                traj_states = []
                traj_actions = []
                traj_pcds = []
                ############# stage {pre-1 and 1-stage} starts #############
                while current_frame < pre_2_frame:
                    traj_actions.append(source_demo["action"][current_frame].copy())
                    traj_states.append(source_demo["state"][current_frame].copy())
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    scene_pcd = np.concatenate([source_pcd, distract_pcd], axis=0)
                    points_xyz = scene_pcd[..., :3]
                    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points, h=3)
                    scene_pcd = scene_pcd[sample_indices]
                    traj_pcds.append(scene_pcd)

                    current_frame += 1
                ############# interpolate 1 #############
                for k in range(interpolate_steps):                    
                    trans = np.array([0, 0, interpolate_step_size * (k+1)])
                    action = source_demo["action"][current_frame].copy()
                    action[:3] += trans
                    traj_actions.append(action)

                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans)
                    
                    scene_pcd = np.concatenate([pcd_obj_robot, pcd_tar, distract_pcd], axis=0)
                    scene_pcd_cropped, _ = self.pcd_divide(scene_pcd, [crop_bbox])
                    points_xyz = scene_pcd_cropped[..., :3]
                    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points, h=3)
                    scene_pcd_cropped = scene_pcd_cropped[sample_indices]
                    traj_pcds.append(scene_pcd_cropped)
                current_frame += 1
                ############# stage {pre2} starts #############
                while current_frame < stage_2_frame:
                    action = source_demo["action"][current_frame].copy()
                    action[:3] += trans
                    traj_actions.append(action)

                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans)
                    
                    scene_pcd = np.concatenate([pcd_obj_robot, pcd_tar, distract_pcd], axis=0)
                    scene_pcd_cropped, _ = self.pcd_divide(scene_pcd, [crop_bbox])
                    points_xyz = scene_pcd_cropped[..., :3]

                    # print(points_xyz.shape)
                    # pcd_visualizer.visualize_pointcloud(points_xyz)

                    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points, h=3)
                    scene_pcd_cropped = scene_pcd_cropped[sample_indices]
                    traj_pcds.append(scene_pcd_cropped)

                    current_frame += 1
                ############## interpolate 2 #############
                for k in range(interpolate_steps):                    
                    trans = np.array([0, 0, interpolate_step_size * interpolate_steps - interpolate_step_size * (k+1)])
                    action = source_demo["action"][current_frame].copy()
                    action[:3] += trans
                    traj_actions.append(action)

                    state = source_demo["state"][current_frame].copy()
                    state[:3] += trans
                    traj_states.append(state)
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    pcd_tar, pcd_obj_robot = self.pcd_divide(source_pcd, [tar_bbox])
                    pcd_obj_robot = self.pcd_translate(pcd_obj_robot, trans)
                    
                    scene_pcd = np.concatenate([pcd_obj_robot, pcd_tar, distract_pcd], axis=0)
                    scene_pcd_cropped, _ = self.pcd_divide(scene_pcd, [crop_bbox])
                    points_xyz = scene_pcd_cropped[..., :3]
                    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points, h=3)
                    scene_pcd_cropped = scene_pcd_cropped[sample_indices]
                    traj_pcds.append(scene_pcd_cropped)
                current_frame += 1
                ############# stage {2} starts #############
                while current_frame < source_demo["state"].shape[0]:
                    traj_actions.append(source_demo["action"][current_frame].copy())
                    traj_states.append(source_demo["state"][current_frame].copy())
                    
                    source_pcd = source_demo["point_cloud"][current_frame].copy()
                    scene_pcd = np.concatenate([source_pcd, distract_pcd], axis=0)
                    points_xyz = scene_pcd[..., :3]
                    sample_indices = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points, h=3)
                    scene_pcd = scene_pcd[sample_indices]
                    traj_pcds.append(scene_pcd)

                    current_frame += 1

                generated_episode = {
                    "state": traj_states, 
                    "action": traj_actions,
                    "point_cloud": traj_pcds,
                }
                generated_episodes.append(generated_episode)

                if video_pcd:
                    print(f"Saving video for episode {i}")
                    self.point_cloud_to_video(generated_episode["point_cloud"], f'/home/zhengrong/projects/Spatial-Generalization/data/video/{exp_name}/{i}_{index}.mp4', elev=20, azim=330)
                ############## end of generating one episode ##############
        
        for i in tqdm(range(self.n_source_episodes)):
            source_episode = self.replay_buffer.get_episode(i)
            generated_episodes.append(source_episode)
        
        # save the generated episodes
        save_episode_dir = f"/home/zhengrong/projects/Spatial-Generalization/data/demo/{exp_name}/{exp_name}_distracted.zarr"
        self.save_episodes(generated_episodes, save_episode_dir)

    @staticmethod
    def point_cloud_to_video(point_clouds, output_file, fps=15, elev=30, azim=45):
        import matplotlib.pyplot as plt
        import imageio
        """
        point_clouds: (N, 6) numpy
        elev: 仰角（度）
        azim: 方位角（度）
        """
        
        fig = plt.figure(figsize=(8, 6), dpi=150)
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
            
            ax.set_xlim(min_vals[0], max_vals[0])
            ax.set_ylim(min_vals[1], max_vals[1])
            ax.set_zlim(min_vals[2], max_vals[2])
            
            # 设置刻度标签格式
            formatter = FormatStrFormatter('%.1f')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            
            ax.grid(True)
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
        import h5py
        save_dir = save_dir.replace('.zarr', '.hdf5')
        with h5py.File(save_dir, 'w') as f:
            f.create_dataset('state', data=state_arrays, compression='gzip')
            f.create_dataset('point_cloud', data=point_cloud_arrays, compression='gzip')
            f.create_dataset('action', data=action_arrays, compression='gzip')
            f.create_dataset('episode_ends', data=episode_ends_arrays, compression='gzip')
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
    exp_name = "bear"
    zarr_path = f"/home/zhengrong/projects/Spatial-Generalization/data/demo/{exp_name}.zarr"
    augmenter = DemoAugmenter(zarr_path)
    # augmenter.crop_distracted_obj()

    # distracted_pcd_list = distracted_pcds(mesh_root="/home/zhengrong/projects/Spatial-Generalization/obj", mesh_list=["ball.obj", "bottle.obj", "cone.obj"])
    distracted_pcd_list = load_crop_obj("/home/zhengrong/projects/Spatial-Generalization/data/demo/distracted")
    gen_demos = augmenter.distracted_object_augment(exp_name=exp_name, distracted_pcd_list=distracted_pcd_list, video_pcd=True)

    
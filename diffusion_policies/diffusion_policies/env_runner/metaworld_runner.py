import imageio
import os
import json
import wandb
import time
import numpy as np
import torch
import collections
import tqdm
from diffusion_policies.env import MetaWorldEnv
from diffusion_policies.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policies.gym_util.multiview_video_recording_wrapper import MultiviewVideoRecordingWrapper

from diffusion_policies.common.pytorch_util import dict_apply
from diffusion_policies.env_runner.base_runner import BaseRunner
from termcolor import cprint
import diffusion_policies.common.logger_util as logger_util

# from multiprocessing import Pool, Queue, Manager
import multiprocessing as mp
import concurrent.futures
from itertools import product

DONE_AFTER_SUCCESS_STEPS = 25	# time for lifting up, env steps


def naive_print(x):
    print(x)


class MetaworldRunner(BaseRunner):
    
    def __init__(self,
                 output_dir,
                 shape_meta: dict,
                 image_obs_only=False,
                 state_obs_only=False,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 use_point_crop=True,
                 device="cuda:0",
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.obs_shape_meta = shape_meta['obs']
        
        if image_obs_only:
            cprint("Skip rendering point clouds!", 'cyan')

        self.image_obs_only = image_obs_only
        self.state_obs_only = state_obs_only
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_steps
        self.eval_episodes = eval_episodes
        self.device = device

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def env_fn(self, render_device):
        return MultiStepWrapper(
            MultiviewVideoRecordingWrapper(
                MetaWorldEnv(task_name=self.task_name, device=render_device,
                                image_obs_only=self.image_obs_only, state_obs_only=self.state_obs_only),),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_episode_steps,
            reward_agg_method='sum',
        )

    def run(self, policy, save_video=False):
        raise NotImplementedError
        device = self.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        
        env = self.env_fn(device)
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # print("epoch:", episode_idx)

            # start rollout
            obs = env.reset()
            policy.reset()

            target_pos = np.round(env.get_target_pos(), 3)
            object_pos = np.round(env.get_object_pos(), 3)

            done = False
            traj_reward = 0
            ep_success = False
            ep_success_times = 0

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # print("obs_dict.keys", obs_dict.keys())
                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    for key in self.obs_shape_meta.keys():
                        obs_dict_input[key] = obs_dict[key].unsqueeze(0)
                    # print("obs_dict_input:", obs_dict_input)    # not normalized
                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env.step(action)

                traj_reward += reward
                done = np.all(done)
                step_success = max(info['success'])
                ep_success_times += step_success
                # if ep_success_times > 0:
                #     print("success times:", ep_success_times)

                if ep_success_times >= DONE_AFTER_SUCCESS_STEPS:
                    done = True
                    ep_success = True
                    break

            all_success_rates.append(ep_success)
            all_traj_rewards.append(traj_reward)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None

        return log_data


    def eval_env_worker(self, policy, gpu_id, cpu_id, eval_idx_this_worker, save_video=False, eval_mode='eval'):
        successes_this_worker = []
        results_this_worker = {}

        device = f"cuda:{gpu_id}"
        policy.to(device)

        env = self.env_fn(device)
        env.env.env.env.reset_mode = eval_mode
        # env.reset()
        cprint(f"Start evaluating {len(eval_idx_this_worker)} episodes on gpu_id:{gpu_id}, cpu_id:{cpu_id} with reset_mode: {eval_mode}", 'blue')

        # time.sleep(5)

        for ep_idx in tqdm.tqdm(eval_idx_this_worker):
            policy.reset()
            # print("ep_idx:", ep_idx)
            obs = env.reset(config_idx=ep_idx)

            # handling skip
            if obs is None:
                continue

            # handling init orientation failure
            ori = obs["full_state"][8:11]
            if np.linalg.norm(ori) > 0.01:
                cprint("Skip init orientation failure", 'red')
                continue


            target_pos = np.round(env.get_target_pos(), 3)
            object_pos = np.round(env.get_object_pos(), 3)

            # print("target_pos:", target_pos)
            # print("object_pos:", object_pos)

            done = False
            ep_success = False
            ep_success_times = 0
            pick_success = False
            pick_success_times = 0
            ep_step = 0

            while not done:
                ep_step += 1
                # print("ep_step:", ep_step)
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    for key in self.obs_shape_meta.keys():
                        obs_dict_input[key] = obs_dict[key].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                # print("action shape:", action.shape)
                obs, reward, done, info = env.step(action)

                done = np.all(done)
                step_success = max(info['success'])
                ep_success_times += step_success
                pick_success = max(info['pick_success']) if 'pick_success' in info else 0
                pick_success_times += pick_success

                if pick_success_times * action.shape[0] >= DONE_AFTER_SUCCESS_STEPS:
                    pick_success = True

                if ep_success_times * action.shape[0] >= DONE_AFTER_SUCCESS_STEPS:
                    done = True
                    ep_success = True
                    break
            
            successes_this_worker.append(ep_success)
            results_this_worker[ep_idx] = {
                    "target_pos": target_pos.tolist(),
                    "object_pos": object_pos.tolist(),
                    "success": ep_success,
                    "pick_success": pick_success,
                }
            
            # print("save_video:", save_video)
            if save_video:
                main_video, wrist_video, wrist2_video, side_video = env.env.get_video()    # (T, C, H, W)
                target_pos_name = f"{target_pos[0]:.2f}-{target_pos[1]:.2f}-{target_pos[2]:.2f}"
                object_pos_name = f"{object_pos[0]:.2f}-{object_pos[1]:.2f}-{object_pos[2]:.2f}"
                success_tag = "success" if ep_success else "fail"
                video_dir = os.path.join(self.output_dir, 'videos')
                os.makedirs(video_dir, exist_ok=True)

                for video, video_name in zip([main_video, wrist_video, wrist2_video, side_video], ["main", "wrist", "wrist2", "side"]):
                    output_video_file = os.path.join(video_dir, f"eval_obj-{object_pos_name}_tar-{target_pos_name}_{success_tag}_{video_name}.mp4")
                    with imageio.get_writer(output_video_file, fps=30) as writer:
                        for frame in video:
                            frame = frame.transpose(1, 2, 0)
                            frame = frame.astype(np.uint8)
                            writer.append_data(frame)
            
            
        return successes_this_worker, results_this_worker


    def eval(self, policy: torch.nn.Module, n_gpu=1, n_cpu_per_gpu=2, save_video=False, eval_mode='eval'):
        cprint(f"Start evaluating {self.eval_episodes} episodes on {n_gpu} GPUs and {n_gpu * n_cpu_per_gpu} CPUs with reset_mode {eval_mode}...", "light_blue")

        eval_idx_per_worker = {}
        for gpu_id, cpu_id in product(range(n_gpu), range(n_cpu_per_gpu)):
            eval_idx_per_worker[(gpu_id, cpu_id)] = []
        
        for i in range(self.eval_episodes):
            gpu_id = i % n_gpu
            cpu_id = (i // n_gpu) % n_cpu_per_gpu
            eval_idx_per_worker[(gpu_id, cpu_id)].append(i)

        with concurrent.futures.ProcessPoolExecutor(n_gpu * n_cpu_per_gpu, mp_context=mp.get_context('spawn')) as executor:
            futures = [executor.submit(self.eval_env_worker, policy, gpu_id, cpu_id, eval_idx_per_worker[(gpu_id, cpu_id)], save_video, eval_mode)
                        for gpu_id, cpu_id in product(range(n_gpu), range(n_cpu_per_gpu))]
            mp_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        all_successes = []
        all_eval_results = {}
        for successes_this_worker, results_this_worker in mp_results:
            all_successes.extend(successes_this_worker)
            all_eval_results.update(results_this_worker)

        total_mean_success_rate = np.around(np.mean(all_successes), 3)
        with open(os.path.join(self.output_dir, f'success_rate_{total_mean_success_rate:.3f}.txt'), 'w') as f:
            f.write(f"{total_mean_success_rate}\n")

        json_filename = os.path.join(self.output_dir, 'eval_results.json')
        with open(json_filename, 'w') as f:
            json.dump(all_eval_results, f)

        log_data = dict()
        log_data['mean_success_rates'] = total_mean_success_rate

        return log_data

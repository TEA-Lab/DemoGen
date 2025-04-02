import imageio
import os
import json
import wandb
import time
import numpy as np
import torch
import collections
import tqdm
from diffusion_policies.env import Robosuite3DEnv
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
import h5py


# TODO: random ee init
INITIAL_ACTIONS = {
    "Coffee": np.array([-0.1137, 0.009355, 0.99702, 2.098, 2.2467, 0.08312, -1]),
    "HammerCleanup": np.array([-0.091492, -0.0070445, 1.0228, 2.1568, 2.1126, 0.17258, -1]),
    # "MugCleanup": np.array([-0.091492, -0.0070445, 1.0228, 2.1568, 2.1126, 0.17258, -1]),
    "Kitchen": np.array([-0.18911, -0.0019802, 1.0137, 2.1753, 2.1105, 0.15635, -1]),
    "ThreePieceAssembly": np.array([-0.11374, 0.0012959, 0.99609, 2.1754, 2.176, 0.098959, -1]),
}



class RobosuiteRunner(BaseRunner):
    
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
        self.source_dataset = None
        self.obs_shape_meta = shape_meta['obs']
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
    
    def prepare_random_init_state(self):
        if self.task_name == "three_piece_assembly":
            demo_index = 3
            self.object_index = {
                "object1": [17, 18, 19],
                "object2": [24, 25, 26],
            }
            self.transform_region = {
                "object1": {
                    "x": (-0.01, 0.14),
                    "y": (-0.21, -0.16)
                },
                "object2": {
                    "x": (-0.3, -0.1),
                    "y": (0.16, 0.21)
                }
            }
        elif self.task_name == "coffee":
            demo_index = 2
            self.object_index = {
                "object1": [10, 11, 12],
            }
            self.transform_region = {
                "object1": {
                    "x": (-0.15, -0.05),
                    "y": (0.15, 0.35)
                },
            }
        elif self.task_name == "hammer_cleanup":
            demo_index = 4
            self.object_index = {
                "object1": [11, 12, 13],
            }
            self.transform_region = {
                "object1": {
                    "x": (-0.20, 0.1),
                    "y": (-0.25, -0.15)
                },
            }
        elif self.task_name == "kitchen":
            demo_index = 0
            self.object_index = {
                "object1": [11, 12, 13],
            }
            self.transform_region = {
                "object1": {
                    "x":  (-0.2, -0.1),
                    "y": (-0.20, -0.10)
                },
            }

        f = h5py.File(self.source_dataset, "r")
        demos = list(f["data"].keys())
        ep = demos[demo_index]
        self.fixed_init_state = f["data/{}/states".format(ep)][0]
        self.initial_state_model = f["data/{}".format(ep)].attrs["model_file"]
        f.close()

    def get_random_init_state(self):
        init_state = dict()
        init_state["model"] = self.initial_state_model
        init_state["states"] = self.fixed_init_state.copy()
        for obj_key in self.object_index.keys():
            x = np.random.uniform(*self.transform_region[obj_key]["x"])
            y = np.random.uniform(*self.transform_region[obj_key]["y"])
            # print("random init:", x, y)
            init_state["states"][self.object_index[obj_key][0]] = x
            init_state["states"][self.object_index[obj_key][1]] = y
        # cprint("init state {}".format(init_state["states"]), "cyan")
        return init_state


    def env_fn(self, render_device):

        return MultiStepWrapper(Robosuite3DEnv(self.source_dataset),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_episode_steps,
            reward_agg_method='sum',
        )

    def eval_env_worker(self, policy, gpu_id, cpu_id, eval_idx_this_worker):
        cprint(f"Start evaluating {len(eval_idx_this_worker)} episodes on gpu_id:{gpu_id}, cpu_id:{cpu_id}", 'blue')
        successes_this_worker = []

        device = f"cuda:{gpu_id}"
        policy.to(device)

        env = self.env_fn(device)
        task_name = env.env.task_name
        init_action = INITIAL_ACTIONS[task_name]
        
        for ep_idx in tqdm.tqdm(eval_idx_this_worker):
            policy.reset()
            random_init_state = self.get_random_init_state()
            env.reset()
            env.reset_to(random_init_state)
            obs, reward, done, info = env.step(np.tile(init_action, (self.n_action_steps, 1)))

            done = False
            ep_success = False
            ep_step = 0

            while not done:
                ep_step += 1
                # print("ep_idx:", ep_idx, "ep_step:", ep_step)
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    for key in self.obs_shape_meta.keys():
                        obs_dict_input[key] = obs_dict[key].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                obs, reward, done, info = env.step(action)
                ep_success = env.check_success() or ep_success

                if ep_success:
                    break
            
            successes_this_worker.append(ep_success)
            
        return successes_this_worker


    def __eval(self, policy: torch.nn.Module, n_gpu=1, n_cpu_per_gpu=2, save_video=False, eval_mode='eval'):
        successes_this_worker = []

        device = f"cuda:0"
        policy.to(device)

        env = self.env_fn(device)
        task_name = env.env.task_name
        init_action = INITIAL_ACTIONS[task_name]
        
        for ep_idx in tqdm.tqdm(range(self.eval_episodes)):
            policy.reset()
            # obs = env.reset()
            obs, reward, done, info = env.step(np.tile(init_action, (self.n_action_steps, 1)))

            done = False
            ep_success = False
            ep_step = 0

            while not done:
                ep_step += 1
                # print("ep_idx:", ep_idx, "ep_step:", ep_step)
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    for key in self.obs_shape_meta.keys():
                        obs_dict_input[key] = obs_dict[key].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)
                obs, reward, done, info = env.step(action)
                ep_success = env.check_success() or ep_success

                if ep_success:
                    break
            
            successes_this_worker.append(ep_success)
        total_mean_success_rate = np.around(np.mean(successes_this_worker), 3)
        with open(os.path.join(self.output_dir, f'success_rate_{total_mean_success_rate:.3f}.txt'), 'w') as f:
            f.write(f"{total_mean_success_rate}\n")

    def eval(self, policy: torch.nn.Module, n_gpu=1, n_cpu_per_gpu=2, save_video=False, eval_mode='eval'):
        self.prepare_random_init_state()

        cprint(f"Start evaluating {self.eval_episodes} episodes on {n_gpu} GPUs and {n_gpu * n_cpu_per_gpu} CPUs", "light_blue")

        eval_idx_per_worker = {}
        for gpu_id, cpu_id in product(range(n_gpu), range(n_cpu_per_gpu)):
            eval_idx_per_worker[(gpu_id, cpu_id)] = []
        
        for i in range(self.eval_episodes):
            gpu_id = i % n_gpu
            cpu_id = (i // n_gpu) % n_cpu_per_gpu
            eval_idx_per_worker[(gpu_id, cpu_id)].append(i)

        with concurrent.futures.ProcessPoolExecutor(n_gpu * n_cpu_per_gpu, mp_context=mp.get_context('spawn')) as executor:
            futures = [executor.submit(self.eval_env_worker, policy, gpu_id, cpu_id, eval_idx_per_worker[(gpu_id, cpu_id)])
                        for gpu_id, cpu_id in product(range(n_gpu), range(n_cpu_per_gpu))]
            mp_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        all_successes = []
        for successes_this_worker in mp_results:
            all_successes.extend(successes_this_worker)

        total_mean_success_rate = np.around(np.mean(all_successes), 3)
        with open(os.path.join(self.output_dir, f'success_rate_{total_mean_success_rate:.3f}.txt'), 'w') as f:
            f.write(f"{total_mean_success_rate}\n")

        log_data = dict()
        log_data['mean_success_rates'] = total_mean_success_rate

        return log_data

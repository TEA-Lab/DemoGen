from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
# from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from termcolor import cprint
from diffusion_policies.common.pytorch_util import dict_apply
from diffusion_policies.dataset.base_dataset import BaseImageDataset
from diffusion_policies.model_dp_umi.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policies.common.sampler import SequenceSampler, get_val_mask
from diffusion_policies.common.normalize_util import get_image_range_normalizer, get_image_identity_normalizer
register_codecs()

class MetaworldImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            zarr_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):

        # replay_buffer = ReplayBuffer.load_zarr(zarr_path)     # save RAM but slower

        cprint('Copying zarr dataset from disk to RAM.', 'green')
        # compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        keys = ['main_img', 'wrist_img', 'agent_pos', 'action']
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys
              )
        # cprint('Loaded!', 'green')

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.affinity_set = False

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        assert self.replay_buffer['action'].shape[-1] == 4
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            assert key == "agent_pos"
            assert self.replay_buffer['agent_pos'].shape[-1] == 4
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer['agent_pos'])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.affinity_set:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([])
            self.affinity_set = True

        # threadpool_limits(1)
        # print("sampler start")
        # import ipdb; ipdb.set_trace()
        data = self.sampler.sample_sequence(idx)    # BUG: <__array_function__ internals>:200: RuntimeWarning: invalid value encountered in cast

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1).astype(np.float32) / 255.     # NOTE: !!!!!!!!!!!!!!!!!!!!!
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

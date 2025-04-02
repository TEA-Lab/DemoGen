from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policies.common.pytorch_util import dict_apply
from diffusion_policies.common.replay_buffer import ReplayBuffer
from diffusion_policies.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policies.model_dp3.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policies.dataset.base_dataset import BasePointcloudDataset

class GalaxeaDataset(BasePointcloudDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['left_state', 'right_state', 'left_action', 'right_action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': np.concatenate([self.replay_buffer['left_action'], self.replay_buffer['right_action']], axis=1),
            'agent_pos': np.concatenate([self.replay_buffer['left_state'], self.replay_buffer['right_state']], axis=1),
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = np.concatenate([sample['left_state'], sample['right_state']], axis=1).astype(np.float32)
        # print(agent_pos.shape)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': np.concatenate([sample['left_action'], sample['right_action']], axis=1).astype(np.float32), # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


import gym
import numpy as np
from termcolor import cprint


class MultiviewVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

        self.main_frames = list()
        self.wrist_frames = list()
        self.wrist2_frames = list()
        self.side_frames = list()

    def _prepare_frames(self):
        main_frame, wrist_frame, wrist2_frame, side_frame = self.env.render(mode=self.mode)
        assert main_frame.dtype == np.uint8
        assert wrist_frame.dtype == np.uint8
        assert wrist2_frame.dtype == np.uint8
        assert side_frame.dtype == np.uint8
        self.main_frames.append(main_frame)
        self.wrist_frames.append(wrist_frame)
        self.wrist2_frames.append(wrist2_frame)
        self.side_frames.append(side_frame)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.main_frames = list()
        self.wrist_frames = list()
        self.wrist2_frames = list()
        self.side_frames = list()

        self._prepare_frames()
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        
        self._prepare_frames()
        
        return result
    
    def get_video(self):
        main_video = np.stack(self.main_frames, axis=0) # (T, H, W, C)
        wrist_video = np.stack(self.wrist_frames, axis=0)
        wrist2_video = np.stack(self.wrist2_frames, axis=0)
        side_video = np.stack(self.side_frames, axis=0)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        main_video = main_video.transpose(0, 3, 1, 2)
        wrist_video = wrist_video.transpose(0, 3, 1, 2)
        wrist2_video = wrist2_video.transpose(0, 3, 1, 2)
        side_video = side_video.transpose(0, 3, 1, 2)
        return main_video, wrist_video, wrist2_video, side_video


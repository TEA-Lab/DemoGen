import time
from multiprocessing.managers import SharedMemoryManager
import zarr
import torch
import numpy as np
import torch
from diffusion_policies.common.pytorch_util import dict_apply
import hydra
from omegaconf import OmegaConf, DictConfig
import pathlib

from utils.keystroke_counter import KeystrokeCounter, Key, KeyCode
from utils.panda_oyhand_env import PandaOYhandEnv
from diffusion_policies.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace

from termcolor import cprint

DIM_ACTION = 12
N_POINTS = 1024
HORIZON = 8
N_OBS = 2
N_ACTIONS = 5
N_ACTIONS_FINISH = 3
MAX_EPISODE_STEPS = 100
OBS_KEYS = ['point_cloud', 'agent_pos']

CKPT_PATH = "data/ckpts/demogen0312-jar-dp3_fast-seed0/checkpoints/6328.ckpt"

@hydra.main(
    #version_base=None,
    config_path="diffusion_policies/diffusion_policies/config",
    config_name="dp3_fast"
)
def main(cfg: OmegaConf):
    env = PandaOYhandEnv(camera=None)
    env.go_home([0.49, -0., 0.44, 3.14, 0, 0])
    
    env = PandaOYhandEnv()
    
    try:
        
        OmegaConf.resolve(cfg)
        cfg.horizon = HORIZON
        cfg.n_obs_steps = N_OBS
        cfg.n_action_steps = N_ACTIONS
        workspace = TrainDiffusionUnetHybridPointcloudWorkspace(cfg)
        model = workspace.model
        latest_ckpt_path = pathlib.Path(CKPT_PATH)

        if latest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {latest_ckpt_path}", 'magenta')
            workspace.load_checkpoint(path=latest_ckpt_path)
        else:
            cprint(f"No checkpoint found at {latest_ckpt_path}", 'magenta')
            exit(0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = model.to(device)
        
        obs = env.reset()
        
        time.sleep(0.1)
        
        obs = env.get_obs()
        init_obs=obs
        init_position=init_obs["agent_pos"]
        
        all_obs_dict = {
            "point_cloud": np.zeros((MAX_EPISODE_STEPS, N_POINTS, 6)),
            "agent_pos": np.zeros((MAX_EPISODE_STEPS, DIM_ACTION))
        }
        all_obs_dict['point_cloud'][0] = obs['point_cloud']
        all_obs_dict['agent_pos'][0] = obs['agent_pos']
        
        all_actions = np.zeros((MAX_EPISODE_STEPS, DIM_ACTION))
        

        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter:
                stop=False
                finish = False
                action_idx = 1
                while not stop and action_idx < MAX_EPISODE_STEPS:
                    print("action_idx:", action_idx)

                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            stop = True
                    
                    # update obs[i]
                    obs = env.get_obs()
                    all_obs_dict['point_cloud'][action_idx] = obs['point_cloud']
                    all_obs_dict['agent_pos'][action_idx] = obs['agent_pos']
                    
                    # update obs[0, 1] -> action[1, 2, 3]; obs[3, 4] -> action[4, 5, 6]
                    if action_idx % N_ACTIONS == 1:
                        if finish:
                            for ii in range(N_ACTIONS_FINISH):
                                action_todo = all_actions[action_idx+ii]
                                obs, reward, done, info = env.step(action_todo)
                            break
                        
                        np_obs_dict = {
                            'point_cloud': all_obs_dict['point_cloud'][action_idx-N_OBS+1:action_idx+1],
                            'agent_pos': all_obs_dict['agent_pos'][action_idx-N_OBS+1:action_idx+1]
                        }
                        # print("arm action:", obs['agent_pos'][:, :6])
                        # print(np_obs_dict['point_cloud'].shape)
                        # print("dict keys:", np_obs_dict.keys())
                        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                        with torch.no_grad():
                            obs_dict_input = {}  # flush unused keys
                            for key in OBS_KEYS:
                                obs_dict_input[key] = obs_dict[key].unsqueeze(0)
                            action_dict = policy.predict_action(obs_dict_input)

                        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                        # action = np_action_dict['action'].squeeze(0)
                        # print("np_action_dict['action'] shape:", np_action_dict['action'].shape)
                        all_actions[action_idx:action_idx+N_ACTIONS] = np.squeeze(np_action_dict['action'])
                        
                        finish = check_finish(np.squeeze(np_action_dict['action']))
                        
                        if finish:
                            last_action = all_actions[action_idx+N_ACTIONS-1].copy()
                            height_offsets = np.zeros((N_ACTIONS_FINISH, 12))
                            height_offsets[:, 2] = np.linspace(0.03, 0.03 * N_ACTIONS_FINISH, N_ACTIONS_FINISH)
                            finish_actions = np.tile(last_action, (N_ACTIONS_FINISH, 1)) + height_offsets
                            finish_actions[:, 5] = -2.4
                            all_actions[action_idx+N_ACTIONS:action_idx+N_ACTIONS+N_ACTIONS_FINISH] = finish_actions
                        
                    action_todo = all_actions[action_idx]
                    if action_todo[2] < 0.17:    # safety
                        action_todo[2] = 0.17
                    obs, reward, done, info = env.step(action_todo)
                    
                    action_idx+=1
    finally:
        env.stop()
                

def check_finish(actions):
    print("yaw:", actions[:, 5])
    print("height:", actions[:, 2])
    print([a[5] < -2.0 for a in actions])
    print([a[2] > 0.35 for a in actions])
    stop = np.sum([a[5] < -2.0 and a[2] > 0.29 for a in actions]) >= 1
    if stop:
        print("Task finished!!!")
    return stop


if __name__=='__main__':
    main()
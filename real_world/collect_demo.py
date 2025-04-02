"""
An interface for collecting demos with keyboard.
"""

import time
import sys
import tty
import termios
import numpy as np
import pickle
import os
import select
import argparse
from utils.panda_oyhand_env import PandaOYhandEnv

from termcolor import cprint

def multiply_list_elements(input_list, multiplier):
    return [element * multiplier for element in input_list]

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        first_char = sys.stdin.read(1)
        timeout = 0.5
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            second_char = sys.stdin.read(1)
            return first_char + second_char
        else:
            return first_char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# save using pickle
def save_state(point_cloud_list, image_list, depth_list, robot_state_list, action_list, save_path):

    point_cloud_arrays = np.stack(point_cloud_list, axis=0)
    image_arrays = np.stack(image_list, axis=0)
    depth_arrays = np.stack(depth_list, axis=0)
    robot_state_arrays = np.stack(robot_state_list, axis=0)
    action_arrays = np.stack(action_list, axis=0)
    
    data = {
        'point_cloud': point_cloud_arrays,
        'image': image_arrays,
        'depth': depth_arrays,
        'agent_pos': robot_state_arrays,
        'action': action_arrays
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("save data to: ", save_path)
    
    return save_path


def collect_demo(save_args):
    env = PandaOYhandEnv()
    env.go_home([0.49, -0., 0.44, 3.14, 0, 0])
    time.sleep(0.1)
    env.reset()
    
    save_base = "data/source_demos"
    
    # current_day_and_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_dir = os.path.join(save_base, save_args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{save_args.traj_name}.pkl")
    
    # to_use_teleop = input("whethert to use teleop:")
    # to_use_arm = input("whether to use arm:")
    to_use_teleop = 'y'
    to_use_arm = 'y'
    if to_use_teleop == 'y':
        to_use_teleop = True
    else:
        to_use_teleop = False
        
    if to_use_arm == 'y':
        to_use_arm = True
    else:
        to_use_arm = False
    
    point_cloud_list = []
    image_list = []
    depth_list = []
    robot_state_list = []
    action_list = []
    
    step_count = 0
    
    # facing back
    # arm_action = [0.601, 0.018, 0.403,  2.794, -1.086, 0]
    
    # facing left
    arm_action = [0.49, 0.01, 0.44, 3.14, 0, 2.4]
    hand_action = [0.64, 3.14, 3.14, 3.14, 3.14, 0]
    
    while True:
        

        cprint("Please input the action: ", "green")

        arm_command = getch()
        if arm_command == 'q':
            save_state(point_cloud_list, image_list, depth_list, robot_state_list, action_list, save_path)
            break

        if arm_command == '`':
            break
        
        step_count += 1
        
        delta_range = 0.03
        delta_angle = 0.15
        
        cprint(f"your arm_command: {arm_command}", "green")
        
        if 'j' in arm_command:
            delta_range = 0.015
        
        if 'w' in arm_command:
            delta_arm_action = [delta_range, 0, 0, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 's' in arm_command:
            delta_arm_action = [-delta_range, 0, 0, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'a' in arm_command:
            delta_arm_action = [0, delta_range, 0, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'd' in arm_command:
            delta_arm_action = [0, -delta_range, 0, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'e' in arm_command:
            delta_arm_action = [0, 0, -delta_range, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'r' in arm_command:
            delta_arm_action = [0, 0, delta_range, 0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if ' ' in arm_command:
            delta_arm_action = [0, 0, 0,  0, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        
        if 'z' in arm_command:
            delta_arm_action = [0, 0, 0,  0, 0, -delta_angle]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'x' in arm_command:
            delta_arm_action = [0, 0, 0,  0, 0, delta_angle]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'c' in arm_command:
            delta_arm_action = [0, 0, 0,  0, -delta_angle, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'v' in arm_command:
            delta_arm_action = [0, 0, 0,  0, delta_angle, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'b' in arm_command:
            delta_arm_action = [0, 0, 0,  -delta_angle, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'n' in arm_command:
            delta_arm_action = [0, 0, 0,  delta_angle, 0, 0]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
            
        if 'k' in arm_command:  # counter-clockwise
            delta_arm_action = [0, 0, 0,  0, 0, -3 * delta_angle]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]
        if 'l' in arm_command:  # clockwise
            delta_arm_action = [0, 0, 0,  0, 0, 3 * delta_angle]
            arm_action = [i+j for i,j in zip(arm_action, delta_arm_action)]

        cprint(f"your arm_action: {arm_action}", "cyan")
        
        if 'p' in arm_command:  # two fingers, hook up
            hand_action = [0.6415830330331155, 2.32076430637686, 2.29004651154176, 1.7694147956718511, 1.7250834326711952, 0.0]
        if 'o' in arm_command:  # two fingers, insert
            hand_action = [0.6415830330331155, 2.619564674318289, 2.5874506160815933, 1.7694147956718511, 1.7250834326711952, 0.0]
        
            
        if 'u' in arm_command:  # thumb back
            hand_action = [0.64, 3.14, 3.14, 3.14, 3.14, 1.57]
        if 'i' in arm_command:  # grasp u-pillow
            hand_action = [0.3, 2.9, 2.9, 2.9, 2.9, 1.57]
        
        if 'y' in arm_command: # five fingers, open & relax
            hand_action = [0.64, 3.14, 3.14, 3.14, 3.14, 0]
            
        action = arm_action + hand_action

        obs_dict, _, _, _ = env.step(np.array(action))

        point_cloud = obs_dict['point_cloud']
        image = obs_dict['image']
        depth = obs_dict['depth']  
        robot_state = obs_dict['agent_pos']

        point_cloud_list.append(point_cloud)
        robot_state_list.append(robot_state)
        image_list.append(image)
        depth_list.append(depth)
        action_list.append(action)
        

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("exp_name", type=str)
    args.add_argument("traj_name", type=str)
    args = args.parse_args()

    collect_demo(args)
    
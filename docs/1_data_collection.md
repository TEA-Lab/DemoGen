# Data Collection (for Your Own Task)

We provide some source demos under the `data/datasets/source` folder. If you only want to get a sense of how 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 works, you can directly start from the provided demos, and jump to [data_generation](./2_data_generation.md). If you want to collect your own data, you can follow the steps below.



## Data Format

Every signle demonstration trajectory (length: `T`) is a dictionary:
- `"point_cloud"`: Array of shape `(T, Np, 6)`, `Np` is the number of point clouds, `6` denotes [x, y, z, r, g, b].
- `"agent_pos"`: Array of shape `(T, Nd)`, `Nd` is the dim of the robot agent's state, e.g. `7` for Panda + Gripper (6d end-effector + 1d gripper); `22` for Panda + Allegro Hand (6d end-effector + 16d hand joints); `12` for Panda + OYHand (6d end-effector + 6d hand joints); `14` for Galaxea R1 (dual: 6d end-effector + 1d gripper). 
- `"action"`: Array of shape `(T, Nd)`. Actions should share the same dim with the robot state.

These dimension informations should be specified in the `shape_meta:` configuration in the `diffusion_policies/diffusion_policies/config/task/<robot>.yaml` file.



## Data Requirements
𝑫𝒆𝒎𝒐𝑮𝒆𝒏 can be applied to various platforms, including bimanual manipulation and dexterous-hand end-effectors. We provide an interface for collecting demos with keyboard for your reference in `real_world/collect_demo.py`.

To facilitate synthetic generation of visual observations, 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 require the access to **3D point clouds**. This asks for preliminary camera calibration of the depth camera. You can follow the procedures in a [note](https://gist.github.com/hshi74/edabc1e9bed6ea988a2abd1308e1cc96) by Haochen Shi. The camera-related parameters should be noted in the beginning of `real_world/utils/pcd_process.py`.

The point cloud we use is projected from **single-view** depth image instead of multi-view, since (1) the calibration process is time-consuming, (2) single-view camera is more practical for mobile platforms, e.g., ego-centric vision on a humanoid.

Like DP3, we recommend the use of RealSense **L515** rather than the more commonly seen D435, because L515 captures higher-quality point clouds, e.g., fewer holes on the object surface, clearer boundaries between objects and background. We add a DBSCAN clustering step to discard the outlier points in the processing pipeline, which we found could effectively improve the quality of point clouds.

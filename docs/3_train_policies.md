# Visuomotor Policies
In the `diffusion_policies` folder, we aggregate multiple implementations of diffusion policies. They can be easily used through the `.yaml` files under the `./diffusion_policies/config` folder. They are used for: (1) training visuomotor policies based on 𝑫𝒆𝒎𝒐𝑮𝒆𝒏-generated datasets; (2) the empirical study for spatial generalization in the paper. 

## 2D Diffusion Policies
The 2D diffusion policies are adapted from the version appearing in the codebase of [UMI](https://github.com/real-stanford/universal_manipulation_interface), since this version supports easy-to-use multi-GPU training via `accelerate`. We provide various types of visual encoders, including from-scratch ResNet and ViT, and pretrained [R3M](https://github.com/facebookresearch/r3m), [CLIP](https://github.com/openai/CLIP), and [DINOV2](https://github.com/facebookresearch/dinov2). 



## 3D Diffusion Policies
The 3D diffusion policies are adapted from the [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy) codebase. We made the following decides that we found could help improve the policy performance:

- **Data Augmentation.** When there are very limited (especially only one) source demonstration, the policy might easily overfit to some certain random noise in the point cloud observations. We found random jittering on the point clouds could help alleviate the overfitting issue. In the config file, random jittering is turned on by setting `policy:se3_augmentation_cfg:jitter: true`.
- **Learning Rate.** We revise the learning rate schedule `training:lr_scheduler` from `cosine` in the original implementation into `constant_with_warmup` since we find this helps stabilize the training process when the dataset size is large.
- **Training Epochs.** When the size of the dataset varies, `#train_epochs` need to be accordingly adjusted to save training time and maximize policy performance. Empirically, we find that the policy performance often converges when the multiplication of the dataset size counted by `#(o,a)-pairs` and `#train_epochs` is around `2e6`. Thus, we adaptively tune `#train_epochs` according to the dataset size, specified by `training:max_train_steps: 2000000` in the config file. 
- **Horizon Lengths.** Another set of hyperparameters that may affect the policy performance is the lengths of horizons. Empirically, we found `horizon: 8, n_obs_steps: 2, n_action_steps: 5` works well for most of the tasks.


## Policy Training
Once the generated dataset is properly placed under `data/datasets/demogen`, you can start training the policy by running the following command:
```bash
cd diffusion_policies
bash train.sh <dataset> <algo> <robot> <seed>
```
where `<robot>` is mainly used to specify the shapes of actions and robot states, which are defined in the `./diffusion_policies/config/task/<robot>.yaml` file. 

Concrete examples: 
```bash
bash train.sh jar dp3 oyhand 0
```


## Policy Evaluation
𝑫𝒆𝒎𝒐𝑮𝒆𝒏 explicitly edits real-world demos to produce synthetic ones. Thus, the policy trained on the 𝑫𝒆𝒎𝒐𝑮𝒆𝒏-generated datasets can be directly deployed and evaluated, without any sim-to-real transfer process. We provide an implementation for your reference in `real_world/evaluate.py`.
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
from diffusion_policies.workspace.base_workspace import BaseWorkspace
from diffusion_policies.policy.diffusion_unet_hybrid_pointcloud_policy import DiffusionUnetHybridPointcloudPolicy
from diffusion_policies.dataset.base_dataset import BasePointcloudDataset
from diffusion_policies.env_runner.base_runner import BaseRunner
# from diffusion_policies.env_runner.robosuite_runner import RobosuiteRunner
from diffusion_policies.common.checkpoint_util import TopKCheckpointManager
from diffusion_policies.common.json_logger import JsonLogger
from diffusion_policies.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policies.model_dp3.diffusion.ema_model import EMAModel
from diffusion_policies.model_dp3.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridPointcloudWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # self.exclude_keys = ['optimizer', 'model']  # when eval, only use ema_model


    def prepare_normalizer(self):
        cfg = self.cfg
        dataset: BasePointcloudDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 40
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost, no validation set
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BasePointcloudDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        # dataset element: {'obs', 'action'}
        # obs: {'point_cloud': (T,512,3), 'imagin_robot': (T,96,7), 'agent_pos': (T,D_pos)}
        assert isinstance(dataset, BasePointcloudDataset), print(f"dataset must be BasePointcloudDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        if cfg.training.max_train_steps is not None:
            cprint(f"max_train_steps: {cfg.training.max_train_steps}", 'light_cyan')
            cfg.training.num_epochs = int(cfg.training.max_train_steps / len(dataset))
            cprint(f"train_epochs: {cfg.training.num_epochs}", 'light_cyan')
            # if cfg.training.num_epochs > 5000:
            #     cprint("max train epochs: 5000", 'red')
            #     cfg.training.num_epochs = 5000
            

        if cfg.training.rollout_every is None:
            rollout_every = int(cfg.training.num_epochs / 5)
        else:
            rollout_every = cfg.training.rollout_every
        rollout_every = 1e9
            
        if cfg.training.checkpoint_every is None:
            checkpoint_every = int(cfg.training.num_epochs / 2)
        else:
            checkpoint_every = cfg.training.checkpoint_every


        # NOTE: set normalizer to the model, not dataset
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        # cfg.logging.name = str(cfg.logging.name)
        # cprint("-----------------------------", "yellow")
        # cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        # cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        # cprint("-----------------------------", "yellow")
        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in tqdm.tqdm(range(cfg.training.num_epochs)):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                    
                        # compute loss
                        t1_1 = time.time()
                        # print("train batch:", batch['obs']['point_cloud'].shape)
                        # print("train batch:", batch)
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        t1_3 = time.time()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_4 = time.time()
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            # wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                
                if (self.epoch % rollout_every) == (rollout_every - 1) and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy, save_video=cfg.training.save_video)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)

                
                    
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, loss_dict = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % checkpoint_every) == (checkpoint_every - 1):
                    self.save_checkpoint(tag=self.epoch)
                    # checkpointing
                    # if cfg.checkpoint.save_last_ckpt:
                    #     self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                        
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                
                self.global_step += 1
                self.epoch += 1
                del step_log

        # stop wandb run
        # wandb_run.finish()


    def eval(self):
        """
        full_test: if True, run full test, otherwise run eval
        """
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)

        # print("normalizer before loading ckpt:", self.model.normalizer)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # print("normalizer after loading ckpt:", self.model.normalizer)
        
        # print(lastest_ckpt_path)
        # import ipdb; ipdb.set_trace()


        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        if isinstance(env_runner, RobosuiteRunner):
            env_runner.source_dataset = cfg.eval.source_dataset

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()

        runner_log = env_runner.eval(policy, n_gpu=cfg.eval.n_gpu, n_cpu_per_gpu=cfg.eval.n_cpu_per_gpu, save_video=cfg.eval.save_video,
                                        eval_mode=cfg.eval.eval_mode if cfg.eval.eval_mode is not None else 'eval')
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
        

@hydra.main(
    #version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):
    workspace = TrainDiffusionUnetHybridPointcloudWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from openpoints.models.backbone import PointNetEncoder, PointNet2Encoder, PointNextEncoder, PointMLPEncoder, DGCNN
import dotmap
from termcolor import cprint

pointnext_modelnet40_cfg = {
    'blocks': [1, 1, 1, 1, 1, 1],
    'strides': [1, 2, 2, 2, 2, 1],
    'width': 64,
    'in_channels': 3,
    'radius': 0.15,
    'radius_scaling': 1.5,
    'sa_layers': 2,
    'sa_use_res': True,
    'nsample': 32,
    'expansion': 4,
    'aggr_args': {
        'feature_type': 'dp_fj',
        'reduction': 'max'
    },
    'group_args': {
        'NAME': 'ballquery',
        'normalize_dp': True
    },
    'conv_args': {
        'order': 'conv-norm-act'
    },
    'act_args': {
        'act': 'relu'
    },
    'norm_args': {
        'norm': 'bn'
    }
}


pointnet2_modelnet40_cfg = {
    'in_channels': 3,
    'width': None,
    'layers': 3,
    'use_res': False,
    'strides': [2, 4, 1],
    'mlps': [
        [[64, 64, 128]],  # stage 1: 96
        [[128, 128, 256]],
        [[256, 512, 1024]]  # stage 4: 1024
    ],
    'radius': [0.2, 0.4, None],
    'num_samples': [32, 64, None],
    'sampler': 'fps',
    'aggr_args': {
        'NAME': 'convpool',
        'feature_type': 'dp_fj',
        'anisotropic': False,
        'reduction': 'max'
    },
    'group_args': {
        'NAME': 'ballquery',
        'use_xyz': True,
        'normalize_dp': False
    },
    'conv_args': {
        'order': 'conv-norm-act'
    },
    'act_args': {
        'act': 'relu'
    },
    'norm_args': {
        'norm': 'bn'
    }
}

class OpenPointsEncoder(nn.Module):
    def __init__(self,
                 pointnet_type='pointnext',
                 in_channels: int=3,
                 out_channels: int=256,
                 pretrained: bool=True,
                 **kwargs):
        super().__init__()
        
        if pointnet_type == 'pointnext':
            ckpt_path = __file__.split('diffusion_policies')[0] + 'pretrained_encoders/pointnext_modelnet40.pth'
            # create encoder
            cfg = dotmap.DotMap(pointnext_modelnet40_cfg)
            self.encoder = PointNextEncoder(**cfg)
            
        elif pointnet_type == 'pointnet2':
            ckpt_path = __file__.split('diffusion_policies')[0] + 'pretrained_encoders/pointnet2_modelnet40.pth'
            # create encoder
            cfg = dotmap.DotMap(pointnet2_modelnet40_cfg)
            self.encoder = PointNet2Encoder(**cfg)
        else:
            raise NotImplementedError
        
        if pretrained:
        
            # load ckpt

            ckpt = torch.load(ckpt_path)['model']
        
            stat_dicts = {}
            # change parameter name and load
            for k, v in ckpt.items():
                if k.startswith('backbone'):
                    stat_dicts[k.replace('backbone.', '')] = v
            info = self.load_state_dict(ckpt, strict=False)
            cprint(f'Loaded {pointnet_type} encoder from {ckpt_path}', 'green')
            cprint(f'Loaded {info}', 'green')  
                
        self.projection = nn.Linear(1024, out_channels)
        
    def forward(self, x):
        """
        x: B, N, 3
        """
        # make x contiguous
        x = x.contiguous()
        feat = self.encoder.forward_cls_feat(x)
        feat = self.projection(feat)
        return feat
        
        
        
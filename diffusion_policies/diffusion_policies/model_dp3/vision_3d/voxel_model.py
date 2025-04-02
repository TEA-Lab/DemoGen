import torch
import MinkowskiEngine as ME
from torch import nn
import math


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(2)
        receptive_field_size = tensor.size(0)
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(
            mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', gain=None):
    fan = _calculate_correct_fan(tensor, mode)
    if gain is None:
        if nonlinearity == 'gelu':
            gain = 1.45
        else:
            gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


class ImpalaVoxelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 batch_norm=False, kernel_size=3, stride=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=3
            )
        )

        if batch_norm:
            self.layers.append(
                ME.MinkowskiBatchNorm(out_channels)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImpalaVoxelResidualBlockPreAct(nn.Module):

    def __init__(self, num_channels,
                 batch_norm=False,
                 act=ME.MinkowskiReLU):
        super().__init__()
        self.layers = nn.ModuleList()
        if batch_norm:
            self.layers.append(ME.MinkowskiBatchNorm(num_channels))
        self.layers.append(act())
        self.layers.append(ImpalaVoxelConvBlock(in_channels=num_channels,
                                                out_channels=num_channels,
                                                batch_norm=False))
        if batch_norm:
            self.layers.append(ME.MinkowskiBatchNorm(num_channels))
        self.layers.append(act())
        self.layers.append(ImpalaVoxelConvBlock(in_channels=num_channels,
                                                out_channels=num_channels,
                                                batch_norm=False))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        new_out = out + x
        return new_out


class ImpalaVoxelCNN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 batch_norm=True,
                 channel_groups=[16, 32, 32, 32],
                 act='relu',
                 no_stride=False,
                 no_pool=False,
                 ):
        super().__init__()
        act_str = act
        if act == 'relu':
            act = ME.MinkowskiReLU
        elif act == 'gelu':
            act = ME.MinkowskiGELU
        else:
            raise NotImplementedError
        res_block = ImpalaVoxelResidualBlockPreAct
        self.convs = nn.ModuleList()
        for ic, ch in enumerate(channel_groups):
            self.convs.append(
                ImpalaVoxelConvBlock(in_channels=in_channels,
                                     out_channels=ch,
                                     batch_norm=batch_norm,
                                     kernel_size=3,
                                     stride=1 if no_stride else 2  # default is 1, here we use 2 to increase the receptive field
                                     )
            )
            if not no_pool:
                if no_stride:
                    self.convs.append(
                        ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3),
                    )
                else:
                    if ic == 0:
                        self.convs.append(ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3))
            self.convs.append(
                res_block(num_channels=ch,
                          batch_norm=batch_norm,
                          act=act)
            )
            self.convs.append(
                res_block(num_channels=ch,
                          batch_norm=batch_norm,
                          act=act)
            )
            in_channels = ch

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()
        self.avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final = ME.MinkowskiLinear(ch * 2, out_channels, bias=True)
        self.weight_initialization(act=act_str)

    def forward(self, x: ME.SparseTensor, return_before_pool=False):
        for layer in self.convs:
            x = layer(x)
        x1 = self.glob_pool(x)
        x2 = self.avg_pool(x)
        xin = ME.cat(x1, x2)
        out = self.final(xin)

        if return_before_pool:
            return out, x
        else:
            return out

    def weight_initialization(self, act):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                kaiming_normal_(m.kernel, mode="fan_out", nonlinearity=act)

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


def get_activation(act='gleu'):
    if act == 'tanh':
        return nn.Tanh
    elif act == 'elu':
        return nn.ELU
    elif act == 'relu':
        return nn.ReLU
    elif act == 'leakyrelu':
        return nn.LeakyReLU
    elif act == 'sigmoid':
        return nn.Sigmoid
    elif act == 'swish':
        return nn.SiLU
    elif act == 'gelu':
        return nn.GELU


class VoxelModel(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 out_features=256,
                 batch_norm=False,
                 act='gelu',
                 channel_groups=[16, 32, 32],
                 in_channels=3,
                 layer_norm=False,
                 no_pool=False,
                 ):
        super().__init__()
        voxel_kwargs = dict(
            in_channels=in_channels,
            out_channels=embed_dim,
            batch_norm=batch_norm,
            act=act,
            channel_groups=channel_groups,
            no_pool=no_pool,
        )
        backbone = ImpalaVoxelCNN

        self.voxel_backbone = backbone(**voxel_kwargs)

        act = get_activation(act)
        in_channels = embed_dim
        if layer_norm:
            self.projector = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_features),
                act()
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(in_channels, out_features),
                act()
            )

    def forward(self, scene_voxels):
        voxel_embed = self.voxel_backbone(scene_voxels)
        voxel_embed = voxel_embed.features
        out = self.projector(voxel_embed)
        return out

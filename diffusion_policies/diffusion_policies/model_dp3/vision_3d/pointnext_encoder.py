import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from termcolor import cprint
from typing import List
import numpy as np
import copy
from easydict import EasyDict as edict

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        group_args:
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(ratio, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        ratio:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = int(ratio * N)
    fps_idx = farthest_point_sample(xyz, S) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    identity = index_points(points, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points, identity
    

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def group(radius, nsample, xyz, points):
    """
    Input:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_points: group data, [B, N, nsample, 3+D]
    """
    B, N, C = xyz.shape
    idx = query_ball_point(radius, nsample, xyz, xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - xyz.view(B, N, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    return new_points


_NORM_LAYER = dict(
    bn1d=nn.BatchNorm1d,
    bn2d=nn.BatchNorm2d,
    bn=nn.BatchNorm2d,
    in2d=nn.InstanceNorm2d, 
    in1d=nn.InstanceNorm1d, 
    gn=nn.GroupNorm,
    syncbn=nn.SyncBatchNorm,
    ln=nn.LayerNorm,    # for tokens
)

def create_norm(norm_args, channels, dimension=None):
    """Build normalization layer.
    Returns:
        nn.Module: Created normalization layer.
    """
    if norm_args is None:
        return None
    if isinstance(norm_args, dict):    
        norm_args = edict(copy.deepcopy(norm_args))
        norm = norm_args.pop('norm', None)
    else:
        norm = norm_args
        norm_args = edict()
    if norm is None:
        return None
    if isinstance(norm, str):
        norm = norm.lower()
        if dimension is not None:
            dimension = str(dimension).lower()
            if dimension not in norm:
                norm += dimension
        assert norm in _NORM_LAYER.keys(), f"input {norm} is not supported"
        norm = _NORM_LAYER[norm]
    return norm(channels, **norm_args)

_ACT_LAYER = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    leakyrelu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
)

def create_act(act_args):
    """Build activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if act_args is None:
        return None
    act_args = copy.deepcopy(act_args)
    
    if isinstance(act_args , str):
        act_args = {"act": act_args}    
    
    act = act_args.pop('act', None)
    if act is None:
        return None

    if isinstance(act, str):
        act = act.lower()
        assert act in _ACT_LAYER.keys(), f"input {act} is not supported"
        act_layer = _ACT_LAYER[act]

    inplace = act_args.pop('inplace', True)

    if act not in ['gelu', 'sigmoid']: # TODO: add others
        return act_layer(inplace=inplace, **act_args)
    else:
        return act_layer(**act_args)

class linearblock(nn.Module):
    def __init__(self, *args, norm_args=None, act_args=None, **kwargs):
        super().__init__()
        in_channels = args[0]
        out_channels = args[1]
        bias = kwargs.pop('bias', True)
        self.norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if self.norm_layer is not None else bias
        self.linear_layer = nn.Linear(*args, bias, **kwargs)
        self.act_layer = create_act(act_args)
    
    def forward(self, x):
        x = self.linear_layer(x)
        if self.norm_layer is not None:
            shape = x.shape
            if len(shape) == 4:
                B, N, S, C = x.shape
                x = x.view(B, N * S, C)
                x = torch.transpose(x, 1, 2)
                x = self.norm_layer(x)
                x = torch.transpose(x, 1, 2)
                x = x.view(B, N, S, C)
            else:
                x = torch.transpose(x, 1, 2)
                x = self.norm_layer(x)
                x = torch.transpose(x, 1, 2)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x
    
class SetAbstraction(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 layers = 2,
                 stride = 1,
                 group_args = {'radius': 0.1, 'nsample': 16},
                 norm_args = {'norm': 'bn1d'},
                 act_args = {'act': 'relu'},
                 use_res = True,
                 is_head = False
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = group_args.radius
        self.nsample = group_args.nsample
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1  # current blocks aggregates all spatial information.
        self.use_res = use_res and not self.all_aggr and not self.is_head
        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
        channels[0] = in_channels + 3 * (not is_head)
        if self.use_res:
            self.skipconv = linearblock(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
        convs = []
        for i in range(len(channels) - 1):
            convs.append(linearblock(channels[i], channels[i + 1],
                                     norm_args=norm_args,
                                     act_args=None if i == len(channels) - 2
                                                      and self.use_res else act_args)
                         )
        self.convs = nn.Sequential(*convs)
        self.act = create_act(act_args)
        self.sec = nn.Sequential(
            nn.Linear(3,32),
            nn.BatchNorm1d(32),
        )
    
    def forward(self, x):
        position, feature = x
        if self.is_head:
            feature = self.convs(feature)
            x = [position, feature]
        else:
            if not self.all_aggr:
                new_position, new_feature, identity = sample_and_group(1. / self.stride ,self.radius, self.nsample, position, feature)
            else:
                new_position, new_feature = sample_and_group_all(position, feature)
            
            if self.use_res:
                identity = self.skipconv(identity)
            
            new_feature = self.convs(new_feature)
            new_feature = torch.max(new_feature, 2)[0]  
            
            if self.use_res:
                new_feature = self.act(new_feature + identity)
            
            x = [new_position, new_feature]
        return x

class LocalAggregation(nn.Module):
    def __init__(self,
                 radius,
                 nsample,
                 channels: List[int],
                 norm_args = {'norm': 'bn1d'},
                 act_args = {'act': 'relu'},
                 last_act = True,
                 ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        channels[0] += 3,
        convs = []
        for i in range(len(channels)-1):
            convs.append(linearblock(channels[i], channels[i+1],
                                            norm_args = norm_args,
                                            act_args  = None if i == (len(channels) - 2) and not last_act else act_args)
                        )
        self.convs = nn.Sequential(*convs)
    
    def forward(self, x):
        position, feature = x
        new_feature = group(self.radius, self.nsample, position, feature)
        new_feature = self.convs(new_feature)
        new_feature = torch,max(new_feature, 2)[0]
        x = [position, new_feature]
        return x



class InvResMlp(nn.Module):
    def __init__(self,
                 in_channels,
                 group_args = None,
                 norm_args = None,
                 act_args = None,
                 expansion = 1,
                 use_res = True,
                 num_posconvs = 2,
                ):
        super().__init__()
        self.radius = group_args.radius
        self.nsample = group_args.nsample
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation(self.radius, self.nsample, [in_channels, in_channels],
                                      norm_args = norm_args, act_args = act_args    if num_posconvs > 0 else None)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        for i in range(len(channels)-1):
            pwconv.append(linearblock(channels[i], channels[i+1],
                                            norm_args = norm_args,
                                            act_args = None if i == len(channels) - 2 else act_args,)
                        )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, x):
        position, feature = x
        identity = feature
        feature = self.convs(position, feature)
        feature = self.pwconv(feature)
        if identity.shape[-1] == feature.shape[-1] and self.use_res:
            feature = feature + identity
        feature = self.act(feature)
        x = [position, feature]
        return x
    

class PointNextEncoderXYZ(nn.Module):
    """Encoder for PointNext
    """
    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 normal_channel: bool=False,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            out_channels (int): feature size of output
            normal_channel (bool): whether to use RGB. Defaults to False.
        """

        super().__init__()
        self.out_channels = out_channels
        self.normal_channel = normal_channel

        if normal_channel:
            assert in_channels == 6, cprint(f"PointNet2EncoderXYZRGB only supports 6 channels, but got {in_channels}", "red")
        else:
            assert in_channels == 3, cprint(f"PointNet2EncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        # Set network hyperparameters
        self.blocks = [1, 1, 1, 1, 1, 1]
        self.strides = [1, 2, 2, 2, 2, 1]
        self.width = 32
        self.in_channels = 3
        self.radius = 0.1
        self.radius_scaling = 2
        self.sa_layers = 2
        self.sa_use_res = True
        self.nsample = 32
        self.nsample_scaling = 1
        self.expansion = 4
        self.act_args = {'act': 'relu'}
        self.norm_args = {'norm': 'bn'}
        self.use_res = True
        self.num_posconvs = 2
        self.radii = self._to_full_list(self.radius, self.radius_scaling)
        self.nsample = self._to_full_list(self.nsample, self.nsample_scaling)


        channels = []
        for stride in self.strides:
            if stride != 1:
                self.width *= 2
            channels.append(self.width)
        channels[-1] = self.out_channels
        
        encoder = []      
        group_args = edict()
        for i in range(len(self.blocks)):      
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                channels[i], self.blocks[i], stride=self.strides[i], group_args=group_args,
                is_head=i == 0 and self.strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        

    
    def _to_full_list(self, param, param_scaling = 1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar, then create a list
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args = group_args,
                                     norm_args = self.norm_args, 
                                     act_args = self.act_args, 
                                     is_head = is_head
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(InvResMlp(self.in_channels,
                                norm_args = self.norm_args,
                                act_args = self.act_args,
                                group_args = group_args,
                                expansion = self.expansion,
                                use_res = self.use_res,
                                num_posconvs=self.num_posconvs
                                ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        B, _, _ = x.shape
        if self.normal_channel:
            position = x[:, :, :3]
            feature = x[:, :, 3:]
        else:
            position = x
            feature = x.clone()
        for i in range(0,len(self.encoder)):
            position, feature = self.encoder[i]([position, feature])
        return feature.view(B, self.out_channels)

if __name__ == "__main__":
    x = torch.rand(50, 512, 3)
    model = PointNextEncoderXYZ()
    y = model(x)
    print(y.shape)


        


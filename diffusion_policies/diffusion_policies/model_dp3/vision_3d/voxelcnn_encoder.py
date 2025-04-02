import torch
import torch.nn as nn
import MinkowskiEngine as ME
from diffusion_policies.model_dp3.vision_3d.voxel_model import VoxelModel

@torch.no_grad()
def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim, sorted=False)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def batched_coordinates_array(coords, device=None):
    if device is None:
        if isinstance(coords, torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"

    N = coords.shape[0] * coords.shape[1]
    flat_coords = coords.reshape(N, 3)
    batch_indices = torch.arange(coords.shape[0], device=device).repeat_interleave(coords.shape[1]).view(-1, 1)
    bcoords = torch.cat((batch_indices, flat_coords), dim=-1)
    return bcoords


def create_input_batch(batch, device="cuda", quantization_size=None,
                       speed_optimized=True, quantization_mode='random'):
    if quantization_size is not None:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    batch["coordinates"] = batch["coordinates"].int()
    if quantization_mode == 'random':
        quantization_mode = ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
    elif quantization_mode == 'avg':
        quantization_mode = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    else:
        raise NotImplementedError
    in_field = ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        quantization_mode=quantization_mode,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED if speed_optimized else ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
        device=device,
    )
    return in_field.sparse()

class VoxelCNN(nn.Module):
    """
    from Tao Chen's Visual Dexterity
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super().__init__()
        self.body_net = VoxelModel(embed_dim=256,
                              out_features=out_channels,
                              batch_norm=True,
                              act='relu',
                              channel_groups=[32, 64, 128, 256],
                              in_channels=1,
                              layer_norm=False)
        self.color_channels = 1 # we do not use color for this network, same as Tao Chen's code
        self.act = nn.GELU()
        

    
    def flat_batch_time(self, x):
        return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

    def unflat_batch_time(self, x, b, t):
        return x.view(b, t, *x.shape[1:])
    
    @torch.no_grad()
    def convert_to_sparse_tensor(self, coords, color=None):
        b = coords.shape[0]
        t = coords.shape[1]
        flat_coords = self.flat_batch_time(coords)


        coordinates_batch = batched_coordinates_array(flat_coords, device=coords.device)
        coordinates_batch, uindices = unique(coordinates_batch, dim=0)
        features_batch = torch.full((coordinates_batch.shape[0], self.color_channels),
                                    0.5, device=coordinates_batch.device)
        batch = {
            "coordinates": coordinates_batch,
            "features": features_batch,
        }
        input_batch_sparse = create_input_batch(batch, device=coords.device,
                                                quantization_size=None,
                                                speed_optimized=True,
                                                quantization_mode='random')
        return input_batch_sparse, b, t


    def forward(self, x):
        """
        x: (B, N, 3)
        """
        coords = x.unsqueeze(1)
        color = None
        x_sparse_tensor, b, t = self.convert_to_sparse_tensor(coords=coords, color=color)
        voxel_features = self.body_net(x_sparse_tensor)
        voxel_features = self.act(voxel_features)
        return voxel_features
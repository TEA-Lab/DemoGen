import numpy as np
import torch

def create_se3_augmentation(cfg):
    se3_augmentation = SE3Augmentation(cfg)
    return se3_augmentation
    

class SE3Augmentation:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # rotation
        self.rotation = cfg.rotation
        self.rotation_angle = cfg.rotation_angle # in degree
        # convert to radian
        self.rotation_range = torch.tensor([3.141592653589793 * self.rotation_angle[0] / 180.0,
                                            3.141592653589793 * self.rotation_angle[1] / 180.0,
                                            3.141592653589793 * self.rotation_angle[2] / 180.0])
        
        # translation
        self.translation = cfg.translation
        self.translation_scale = cfg.translation_scale
        
        # jitter
        self.jitter = cfg.jitter
        self.jitter_scale = cfg.jitter_scale
        
        
    def __call__(self, points:torch.Tensor):
        assert isinstance(points, torch.Tensor), "Input points must be a torch.Tensor."
        assert points.dim() == 3, "Input points must be a 3D tensor."
        
        B, N, _ = points.shape  # Batch size, Number of points

        rotation_range = self.rotation_range
        translation_scale = self.translation_scale
        jitter_scale = self.jitter_scale

        if self.rotation:
            x_angle = torch.rand(1) * 2 * rotation_range[0] - rotation_range[0]
            y_angle = torch.rand(1) * 2 * rotation_range[1] - rotation_range[1]
            z_angle = torch.rand(1) * 2 * rotation_range[2] - rotation_range[2]

            # Generate rotation matrix
            Rx = torch.tensor([[1., 0., 0.],
                            [0., torch.cos(x_angle), -torch.sin(x_angle)],
                            [0., torch.sin(x_angle), torch.cos(x_angle)]])

            Ry = torch.tensor([[torch.cos(y_angle), 0., torch.sin(y_angle)],
                            [0., 1., 0.],
                            [-torch.sin(y_angle), 0., torch.cos(y_angle)]])

            Rz = torch.tensor([[torch.cos(z_angle), -torch.sin(z_angle), 0.],
                            [torch.sin(z_angle), torch.cos(z_angle), 0.],
                            [0., 0., 1.]])

            R = torch.mm(Rz, torch.mm(Ry, Rx)).to(points.device)

            # Apply the same rotation to all batches
            points = torch.matmul(points, R.T)

        if self.translation:
            # Generate random translations for each batch
            translations = torch.rand((B, 1, 3)) * 2 * translation_scale - translation_scale
            translations[:, :, 2] = 0.0  # No translation in z-axis
            translations = translations.to(points.device)

            # Apply translations
            points += translations

        if self.jitter:
            # Generate random jitters for each point
            jitters = torch.rand((B, N, 3)) * 2 * jitter_scale - jitter_scale
            jitters = jitters.to(points.device)

            # Apply jitters
            points += jitters

        return points


        

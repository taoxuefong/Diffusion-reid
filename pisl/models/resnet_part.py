from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

__all__ = ['ResNetPart', 'resnet18part', 'resnet34part', 'resnet50part', 'resnet101part',
           'resnet152part']

    
class PatchGenerator(nn.Module):
    """
    Patch Generator for Spatial Transformer Network (STN)
    Generates transformation parameters (theta) for local region extraction
    """
    def __init__(self):
        super(PatchGenerator, self).__init__()

        # Localization network for feature extraction
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Fully connected layers for transformation parameter prediction
        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 2 * 3),  # Output 3x2x3 transformation matrices
        )
        
        # Initialize transformation parameters for 3 local regions
        # Each region has 3x2 parameters (scale_x, scale_y, translation_x, scale_x, scale_y, translation_y)
        path_postion = [1, 1/3, 1/3,
                        1, 1/3, 1/3,
                        1, 1/3, 1/3,
                        1, 1/3, 1/3,
                        1, 1/3, 1/3,
                        1, 1/3, 1/3,]
        
        # Initialize weights and biases
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))

        # Initialize network parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass to generate transformation parameters
        Args:
            x: Input feature map
        Returns:
            theta: Transformation parameters for local region extraction
        """
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1,1))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 2, 3)
        return theta

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for diffusion process
    Args:
        timesteps: Number of diffusion steps
        s: Small constant to prevent beta from being too small
    Returns:
        betas: Noise schedule parameters
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionPatchGenerator(nn.Module):
    """
    Diffusion-based Patch Generator
    Uses diffusion model to generate high-quality transformation parameters
    """
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        super(DiffusionPatchGenerator, self).__init__()
        self.num_steps = num_steps
        self.device = torch.device('cuda')
        
        # Initialize diffusion process parameters
        betas = cosine_beta_schedule(num_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Register buffers for diffusion process
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Compute coefficients for diffusion process
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Compute posterior distribution coefficients
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # Define network structure for noise prediction
        input_dim = 2048 + 512  # Feature dimension + time encoding dimension
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 2 * 3)  # Output theta parameters
        )
        
        # Initialize transformation parameters for pedestrian key regions
        path_postion = [
            # Head region - maintain original size, precise upward shift
            1.0, 1.0, 0.0,    # First row maintain original size
            1.0, 1.0, -0.1,   # Second row slight upward shift
            
            # Upper body region - maintain original size, precise centering
            1.0, 1.0, 0.0,    # First row maintain original size
            1.0, 1.0, 0.0,    # Second row maintain original size
            
            # Lower body region - maintain original size, precise downward shift
            1.0, 1.0, 0.0,    # First row maintain original size
            1.0, 1.0, 0.1,    # Second row slight downward shift
        ]
        
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float32))
        
        # Diffusion model parameters
        self.sampling_timesteps = num_steps
        self.is_ddim_sampling = self.sampling_timesteps < num_steps
        self.ddim_sampling_eta = 1. #can set 0.0 for pure DDIM
        self.self_condition = False
        self.scale = 1.0  # Signal-to-noise ratio scaling factor
        self.theta_renewal = True  # Whether to update low-quality theta, can set to False for faster sampling
        self.use_ensemble = True  # Whether to use ensemble sampling
        
    def extract(self, a, t, x_shape):
        """Extract the appropriate t index for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start, dtype=torch.float32)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x0"""
        return (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
               self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
               
    def model_predictions(self, x, t, x_self_cond=None):
        """Model predictions"""
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        
        # Predict noise and x0
        noise_pred = self.net(x)
        x_start = self.predict_noise_from_start(x, t, noise_pred)
        
        return noise_pred, x_start
        
    def get_timestep_embedding(self, t):
        """Time encoding"""
        half_dim = 512 // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        # Ensure correct t dimension [B]
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)  # [B, 512]
        return emb
        
    def forward(self, x, t, theta=None):
        """Forward pass"""
        # Ensure correct t dimension
        if isinstance(t, int):
            t = torch.tensor([t], device=x.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.size(0) == 1:
            t = t.expand(x.size(0))
            
        # Add time encoding
        t_emb = self.get_timestep_embedding(t)  # [B, 512]
        x = torch.cat([x, t_emb], dim=1)  # [B, 2048+512]
        
        if theta is not None:
            # Training phase: add noise to ground truth theta
            noise = torch.randn_like(theta, dtype=torch.float32)
            noisy_theta = self.q_sample(x_start=theta, t=t, noise=noise)
            noise_pred = self.net(x)
            return noise_pred, noisy_theta, noise
        else:
            # Inference phase: return predicted noise
            noise_pred = self.net(x)
            return noise_pred
            
    def prepare_diffusion_concat(self, gt_theta):
        """Prepare diffusion training target"""
        t = torch.randint(0, self.num_steps, (1,), device=self.device).long()
        
        # If gt_theta is a list, convert it to tensor
        if isinstance(gt_theta, list):
            gt_theta = torch.stack(gt_theta)
            
        # Ensure gt_theta is 4D tensor [B, 3, 2, 3]
        if gt_theta.dim() == 3:
            gt_theta = gt_theta.unsqueeze(0)
            
        # Generate noise
        noise = torch.randn_like(gt_theta, dtype=torch.float32)
        
        # Convert theta to diffusion model input form
        gt_theta = (gt_theta * 2. - 1.) * self.scale
        
        # Add noise
        x = self.q_sample(x_start=gt_theta, t=t, noise=noise)
        
        # Clip and normalize
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.
        
        return x, noise, t
        
    @torch.no_grad()
    def ddim_sample(self, x, num_steps=None, theta_init=None):
        """DDIM sampling"""
        if num_steps is None:
            num_steps = self.num_steps
            
        # Initialize theta parameters
        if theta_init is not None:
            theta = theta_init.view(theta_init.size(0), -1)
        else:
            theta = torch.randn(x.size(0), 3 * 2 * 3, device=x.device, dtype=torch.float32)
            
        # Step-by-step denoising
        ensemble_thetas = []
        for t in reversed(range(num_steps)):
            t_batch = torch.full((x.size(0),), t, device=x.device, dtype=torch.float32)
            
            # Add time encoding
            t_emb = self.get_timestep_embedding(t_batch)  # [B, 512]
            x_t = torch.cat([x, t_emb], dim=1)  # [B, 2048+512]
            
            # Predict noise
            noise_pred = self.net(x_t)
            
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod_prev[t]
            
            # DDIM update step
            if t > 0:
                noise = torch.randn_like(theta, dtype=torch.float32)
            else:
                noise = 0
                
            # Compute sigma and c
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_prev) * (1 - alpha_prev) / (1 - alpha)).sqrt()
            c = (1 - alpha_prev - sigma ** 2).sqrt()
            
            # Update theta
            theta = torch.sqrt(alpha_prev) * theta + \
                   c * noise_pred + \
                   sigma * noise
                   
            # Add theta consistency constraint in final steps
            if t < 100 and theta_init is not None:
                theta = 0.9 * theta + 0.1 * theta_init.view(theta_init.size(0), -1)
                
            # If using theta renewal mechanism
            if self.theta_renewal and t > 0:
                # Compute theta quality scores
                quality_scores = torch.sigmoid(self.net(x_t).mean(dim=1))  # Average for each sample
                threshold = 0.5
                keep_idx = quality_scores > threshold
                
                # Update low-quality theta
                if not keep_idx.all():
                    num_remain = keep_idx.sum()
                    if num_remain > 0:  # Ensure at least one theta is kept
                        theta = theta[keep_idx]
                        # Supplement with new random theta
                        new_theta = torch.randn(x.size(0) - num_remain, 3 * 2 * 3, device=x.device, dtype=torch.float32)
                        theta = torch.cat([theta, new_theta], dim=0)
                    else:  # If no theta is kept, regenerate all theta
                        theta = torch.randn(x.size(0), 3 * 2 * 3, device=x.device, dtype=torch.float32)
                
            # If using ensemble sampling
            if self.use_ensemble and t % 100 == 0:
                ensemble_thetas.append(theta.clone())
                
        # If using ensemble sampling, combine all sampling results
        if self.use_ensemble and ensemble_thetas:
            theta = torch.stack(ensemble_thetas).mean(0)
            
        return theta.view(-1, 3, 2, 3)  # Reshape to correct shape

class ResNetPart(nn.Module):
    """
    ResNet with part features by uniform partitioning
    Combines global and local features for person re-identification
    """
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_parts=3, num_classes=0):
        super(ResNetPart, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        
        # Construct base (pretrained) resnet
        if depth not in ResNetPart.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetPart.__factory[depth](pretrained=pretrained)
        
        # Modify stride in last layer for better feature resolution
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.num_parts = num_parts
        self.num_classes = num_classes
        
        # Base network for feature extraction
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
            
        # Global and local feature pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.rap = nn.AdaptiveAvgPool2d((self.num_parts, 1))  # Regional average pooling

        # Global feature classifier
        self.bnneck = nn.BatchNorm1d(2048)
        init.constant_(self.bnneck.weight, 1)
        init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        # Part feature classifiers
        for i in range(self.num_parts):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(2048))
            init.constant_(getattr(self, name).weight, 1)
            init.constant_(getattr(self, name).bias, 0)
            getattr(self, name).bias.requires_grad_(False)

            name = 'classifier' + str(i)
            setattr(self, name, nn.Linear(2048, self.num_classes, bias=False))
            
        # Initialize patch generation modules
        self.patch_proposal = PatchGenerator()
        self.diffusion_patch = DiffusionPatchGenerator()
        
        if not pretrained:
            self.reset_params()
    
    def forward(self, x):
        x = self.base(x)
        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        f_g = self.bnneck(f_g)
        
        if self.training is False:
            f_g = F.normalize(f_g)
            return f_g

        logits_g = self.classifier(f_g)
        
        if self.training:
            # Training phase: use STN to get initial theta and add noise
            initial_theta = self.patch_proposal(x)  # x is already feature map
            # Prepare diffusion training target
            noisy_theta, noise, t = self.diffusion_patch.prepare_diffusion_concat(initial_theta)
            # Predict noise
            noise_pred = self.diffusion_patch.forward(f_g, t)
            
            # Generate patch features using noisy theta
            f_p = []
            for i in range(3):
                stripe = noisy_theta[:, i, :, :].float()  # Ensure data type is float
                grid = F.affine_grid(stripe, x.size())
                f_p.append(F.grid_sample(x, grid))
        else:
            # Inference phase: use diffusion model to generate optimized theta
            theta = self.diffusion_patch.ddim_sample(f_g)
            
            # Generate patch features using optimized theta
            f_p = []
            for i in range(3):
                stripe = theta[:, i, :, :].float()  # Ensure data type is float
                grid = F.affine_grid(stripe, x.size())
                f_p.append(F.grid_sample(x, grid))
            
        logits_p = []
        fs_p = []
        
        for i in range(self.num_parts):
            f_p_i = self.gap(f_p[i])
            f_p_i = f_p_i.view(f_p[i].size(0), -1)
            f_p_i = self.bnneck(f_p_i)
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
            logits_p_i = getattr(self, 'classifier' + str(i))(f_p_i)
            logits_p.append(logits_p_i)
            
        fs_p = torch.stack(fs_p, dim=-1)
        logits_p = torch.stack(logits_p, dim=-1)

        if self.training:
            return f_g, fs_p, logits_g, logits_p, noise_pred, noisy_theta, noise
        else:
            return f_g, fs_p, logits_g, logits_p

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def extract_all_features(self, x):
        x = self.base(x)
        f_g = self.gap(x)
        f_g = f_g.view(x.size(0), -1)
        f_g = self.bnneck(f_g)
        f_g = F.normalize(f_g)
        
        # Get initial theta
        initial_theta = self.patch_proposal(x)
        
        # Use diffusion model to generate optimized theta
        theta = self.diffusion_patch.ddim_sample(f_g)
        
        # Generate patch features using optimized theta
        f_p = []
        for i in range(self.num_parts):
            stripe = theta[:, i, :, :]
            grid = F.affine_grid(stripe, x.size())
            f_p_i = F.grid_sample(x, grid)
            f_p_i = self.gap(f_p_i)
            f_p_i = f_p_i.view(f_p_i.size(0), -1)
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            f_p_i = F.normalize(f_p_i)
            f_p.append(f_p_i)
            
        fs_p = torch.stack(f_p, dim=-1)
        
        return f_g, fs_p
        

def resnet18part(**kwargs):
    return ResNetPart(18, **kwargs)


def resnet34part(**kwargs):
    return ResNetPart(34, **kwargs)


def resnet50part(**kwargs):
    return ResNetPart(50, **kwargs)


def resnet101part(**kwargs):
    return ResNetPart(101, **kwargs)


def resnet152part(**kwargs):
    return ResNetPart(152, **kwargs)

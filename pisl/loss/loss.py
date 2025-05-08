import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class PSC_LS(nn.Module):
    """Label smoothing loss for patch semantic consistency"""
    def __init__(self):
        super(PSC_LS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, consistency_scores):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uniform_dist = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        ce_loss = (- targets * log_preds).sum(1)
        kld_loss = F.kl_div(log_preds, uniform_dist, reduction='none').sum(1)
        total_loss = (consistency_scores * ce_loss + (1-consistency_scores) * kld_loss).mean()
        return total_loss


class PSC_LR(nn.Module):
    """Label refinement loss for patch semantic consistency"""
    def __init__(self, lam=0.5):
        super(PSC_LR, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lam = lam

    def forward(self, global_logits, part_logits, targets, consistency_scores):
        targets = torch.zeros_like(global_logits).scatter_(1, targets.unsqueeze(1), 1)
        weights = torch.softmax(consistency_scores, dim=1)  # B * P
        weights = torch.unsqueeze(weights, 1)  # B * 1 * P
        part_preds = self.softmax(part_logits)  # B * C * P
        ensemble_preds = (part_preds * weights).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensemble_preds

        log_global_preds = self.logsoftmax(global_logits)
        loss = (-refined_targets * log_global_preds).sum(1).mean()
        return loss


class CameraContrast(nn.Module):
    """
    Camera-aware contrastive loss for cross-camera person re-identification
    """
    def __init__(self, feat_dim, num_cams, num_hards=50, temp=0.07):
        super(CameraContrast, self).__init__()
        self.feat_dim = feat_dim  # D
        self.num_cams = num_cams  # N
        self.num_hards = num_hards
        self.temp = temp
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.register_buffer('proxy', torch.zeros(num_cams, feat_dim))
        self.register_buffer('pids', torch.zeros(num_cams).long())
        self.register_buffer('cids', torch.zeros(num_cams).long())

    def forward(self, features, targets, cams):
        batch_size, feat_dim = features.shape
        features = F.normalize(features, dim=1).cuda()  # B * D
        similarities = features @ self.proxy.T  # B * N
        similarities /= self.temp
        temp_similarities = similarities.detach().clone()
        
        loss = torch.tensor([0.]).cuda()
        for i in range(batch_size):
            pos_mask = (targets[i] == self.pids).float() * (cams[i] != self.cids).float()
            neg_mask = (targets[i] != self.pids).float()
            pos_idx = torch.nonzero(pos_mask > 0).squeeze(-1)
            if len(pos_idx) == 0:
                continue
            hard_neg_idx = torch.sort(temp_similarities[i] + (-9999999.) * (1.-neg_mask), descending=True).indices[:self.num_hards]
            batch_similarities = similarities[i, torch.cat([pos_idx, hard_neg_idx])]
            batch_targets = torch.zeros(len(batch_similarities)).cuda()
            batch_targets[:len(pos_idx)] = 1.0 / len(pos_idx)
            loss += - (batch_targets * self.logsoftmax(batch_similarities)).sum()

        loss /= batch_size
        return loss


class DiffusionThetaLoss(nn.Module):
    """Diffusion model theta consistency loss"""
    def __init__(self, lam_consistency=0.5, lam_contrastive=0.3, temperature=0.07):
        super(DiffusionThetaLoss, self).__init__()
        self.lam_consistency = lam_consistency
        self.lam_contrastive = lam_contrastive
        self.temperature = temperature

    def forward(self, theta_pred, theta_init, noise_pred, noise, features, targets):
        # Ensure dimension matching
        noise_pred = noise_pred.view(noise_pred.size(0), 3, 2, 3)  # [B, 3, 2, 3]
        noise = noise.view(noise.size(0), 3, 2, 3)  # [B, 3, 2, 3]
        
        # Compute noise prediction loss
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Compute theta consistency loss
        theta_pred_flat = theta_pred.view(theta_pred.size(0), -1)
        theta_init_flat = theta_init.view(theta_init.size(0), -1)
        
        # L2 distance loss
        l2_loss = F.mse_loss(theta_pred_flat, theta_init_flat)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(theta_pred_flat, theta_init_flat, dim=1)
        cos_loss = 1 - cos_sim.mean()
        
        # Combine theta consistency loss
        consistency_loss = l2_loss + 0.1 * cos_loss
        
        # Compute contrastive loss for each part
        part_losses = []
        part_weights = []
        
        for i in range(features.size(-1)):
            part_features = F.normalize(features[:, :, i], p=2, dim=1)
            sim_matrix = torch.matmul(part_features, part_features.t()) / self.temperature
            labels = targets.unsqueeze(1) == targets.unsqueeze(0)
            labels = labels.float()
            exp_sim = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
            part_loss = -((log_prob * labels).sum(1) / (labels.sum(1) + 1e-6)).mean()
            
            # Compute weight based on discriminability
            part_weight = torch.exp(-part_loss)
            part_losses.append(part_loss)
            part_weights.append(part_weight)
        
        # Normalize weights
        part_weights = torch.stack(part_weights)
        part_weights = part_weights / part_weights.sum()
        
        # Compute weighted average contrastive loss
        contrastive_loss = sum(w * l for w, l in zip(part_weights, part_losses))
        
        # Total loss
        total_loss = noise_loss + \
                    self.lam_consistency * consistency_loss + \
                    self.lam_contrastive * contrastive_loss
        
        return total_loss, {
            'noise_loss': noise_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'part_weights': [w.item() for w in part_weights]
        }


class DiffusionThetaLoss2(nn.Module):
    """Diffusion model theta consistency loss using average patch features as one choose"""
    def __init__(self, lam_consistency=0.5, lam_contrastive=0.3, temperature=0.07):
        super(DiffusionThetaLoss, self).__init__()
        self.lam_consistency = lam_consistency
        self.lam_contrastive = lam_contrastive
        self.temperature = temperature
        
    def forward(self, theta_pred, theta_init, noise_pred, noise, features, targets):
        # Ensure dimension matching
        noise_pred = noise_pred.view(noise_pred.size(0), 3, 2, 3)  # [B, 3, 2, 3]
        noise = noise.view(noise.size(0), 3, 2, 3)  # [B, 3, 2, 3]
        
        # Compute noise prediction loss
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Compute theta consistency loss
        theta_pred_flat = theta_pred.view(theta_pred.size(0), -1)
        theta_init_flat = theta_init.view(theta_init.size(0), -1)
        
        # L2 distance loss
        l2_loss = F.mse_loss(theta_pred_flat, theta_init_flat)
        
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(theta_pred_flat, theta_init_flat, dim=1)
        cos_loss = 1 - cos_sim.mean()
        
        # Combine theta consistency loss
        consistency_loss = l2_loss + 0.1 * cos_loss
        
        # Compute contrastive loss
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(features_norm, features_norm.t()) / self.temperature
        
        # Construct label matrix: 1 for same-class samples, 0 for different-class samples
        labels = targets.unsqueeze(1) == targets.unsqueeze(0)
        labels = labels.float()
        
        # Compute contrastive loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (log_prob * labels).sum(1) / (labels.sum(1) + 1e-6)
        contrastive_loss = -mean_log_prob.mean()
        
        # Total loss
        loss = noise_loss + \
               self.lam_consistency * consistency_loss + \
               self.lam_contrastive * contrastive_loss
               
        return loss, {
            'noise_loss': noise_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'contrastive_loss': contrastive_loss.item()
        }

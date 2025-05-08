from __future__ import print_function, absolute_import
import time

from .evaluation_metrics import accuracy
from .loss import PSC_LS, PSC_LR, SoftTripletLoss, CrossEntropyLabelSmooth, DiffusionThetaLoss
from .utils.meters import AverageMeter


class PISLTrainer(object):
    """Trainer for PISL model with diffusion model
    
    This trainer implements the training process for the PISL model, which includes:
    1. Global and part feature learning
    2. Patch semantic consistency learning
    3. Diffusion model training for patch generation
    
    Args:
        model: PISL model containing backbone, part generator and diffusion model
        score: Semantic consistency scores between global and part features
        num_class: Number of classes for classification
        num_part: Number of parts to be generated
        Wref: Weight for label refinement in PSC_LR loss
        se: Starting epoch for label smoothing
        Wdiff: Weight for diffusion loss
    """
    def __init__(self, model, score, num_class=500, num_part=6, Wref=0.5, se=5, Wdiff=0.1):
        super(PISLTrainer, self).__init__()
        self.model = model
        self.score = score

        self.num_class = num_class
        self.num_part = num_part
        self.se = se
        self.Wdiff = Wdiff

        # Initialize loss functions
        self.criterion_psclr = PSC_LR(lam=Wref).cuda()  # Patch Semantic Consistency Label Refinement loss
        self.criterion_pscls = PSC_LS().cuda()  # Patch Semantic Consistency Label Smoothing loss
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()  # Cross entropy loss with label smoothing
        self.criterion_tri = SoftTripletLoss().cuda()  # Soft triplet loss for metric learning
        self.criterion_diff = DiffusionThetaLoss().cuda()  # Diffusion model loss

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        """Training process for one epoch
        
        The training process includes:
        1. Feature extraction and patch generation
        2. Loss computation for different components
        3. Model update and statistics recording
        
        Args:
            epoch: Current epoch number
            train_dataloader: Training data loader
            optimizer: Optimizer for model update
            print_freq: Frequency of printing training status
            train_iters: Number of iterations per epoch
        """
        self.model.train()

        # Initialize metrics for monitoring training progress
        batch_time = AverageMeter()  # Time for processing each batch
        losses_gce = AverageMeter()  # Global classification loss
        losses_tri = AverageMeter()  # Triplet loss
        losses_pce = AverageMeter()  # Part classification loss
        losses_diff = AverageMeter()  # Diffusion model loss
        precisions = AverageMeter()  # Classification accuracy

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            # Load and parse data
            data = train_dataloader.next()
            inputs, targets, ca = self._parse_data(data)

            # Forward pass through the model
            # emb_g: global features, emb_p: part features
            # logits_g: global classification logits, logits_p: part classification logits
            # theta_pred: predicted transformation parameters
            # theta_init: initial transformation parameters
            # noise_pred: predicted noise, noise: ground truth noise
            emb_g, emb_p, logits_g, logits_p, theta_pred, theta_init, noise_pred, noise = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # Compute various losses
            loss_gce = self.criterion_psclr(logits_g, logits_p, targets, ca)  # Global classification loss
            loss_tri = self.criterion_tri(emb_g, targets)  # Triplet loss for metric learning

            # Compute part classification loss
            loss_pce = 0.
            if self.num_part > 0:
                if epoch >= self.se:
                    # Use label smoothing after starting epoch
                    for part in range(self.num_part):
                        loss_pce += self.criterion_pscls(logits_p[:, :, part], targets, ca[:, part])
                else:
                    # Use standard cross entropy before starting epoch
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(logits_p[:, :, part], targets)
                loss_pce /= self.num_part

            # Compute diffusion model loss
            loss_diff, _ = self.criterion_diff(theta_pred, theta_init, noise_pred, noise, emb_p, targets)

            # Combine all losses
            loss = loss_gce + loss_tri + loss_pce + self.Wdiff * loss_diff

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record statistics
            prec, = accuracy(logits_g.data, targets.data)

            # Update metrics
            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_pce.update(loss_pce.item())
            losses_diff.update(loss_diff.item())
            precisions.update(prec[0])

            # Update time statistics
            batch_time.update(time.time() - end)
            end = time.time()

            # Print training status
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_DIFF {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_diff.val, losses_diff.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        """Parse input data from data loader
        
        Args:
            inputs: Input data tuple containing:
                - imgs: Input images
                - _: Unused data
                - pids: Person IDs
                - _: Unused data
                - idxs: Sample indices
            
        Returns:
            imgs: Input images on GPU
            pids: Person IDs on GPU
            ca: Semantic consistency scores on GPU
        """
        imgs, _, pids, _, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), ca.cuda()


class PISLTrainerCAM(object):
    """Trainer for PISL model with camera-aware learning and diffusion model
    
    This trainer extends PISLTrainer by adding:
    1. Camera-aware contrastive learning
    2. Camera-aware memory bank
    3. Enhanced diffusion model training
    
    Args:
        model: PISL model
        score: Semantic consistency scores
        memory: Camera-aware memory bank for global features
        memory_p: Camera-aware memory banks for part features
        num_class: Number of classes
        num_part: Number of parts
        Wref: Weight for label refinement
        se: Starting epoch for label smoothing
        Wcam: Weight for camera-aware loss
        Wdiff: Weight for diffusion loss
    """
    def __init__(self, model, score, memory, memory_p, num_class=500, num_part=6, Wref=0.5, se=5, Wcam=0.5, Wdiff=0.1):
        super(PISLTrainerCAM, self).__init__()
        self.model = model
        self.score = score
        self.memory = memory  # Camera-aware memory bank for global features
        self.memory_p = memory_p  # Camera-aware memory banks for part features

        self.num_class = num_class
        self.num_part = num_part
        self.Wcam = Wcam  # Weight for camera-aware loss
        self.Wdiff = Wdiff  # Weight for diffusion loss
        self.se = se  # Starting epoch for label smoothing

        # Initialize loss functions
        self.criterion_psclr = PSC_LR(lam=Wref).cuda()  # Patch Semantic Consistency Label Refinement loss
        self.criterion_pscls = PSC_LS().cuda()  # Patch Semantic Consistency Label Smoothing loss
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()  # Cross entropy loss with label smoothing
        self.criterion_tri = SoftTripletLoss().cuda()  # Soft triplet loss for metric learning
        self.criterion_diffusion = DiffusionThetaLoss(
            lam_consistency=0.5,  # Weight for consistency loss
            lam_contrastive=0.3,  # Weight for contrastive loss
            temperature=0.07  # Temperature parameter for contrastive learning
        ).cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        """Training process for one epoch with camera-aware learning
        
        The training process includes:
        1. Feature extraction and patch generation
        2. Camera-aware contrastive learning
        3. Loss computation for different components
        4. Model update and statistics recording
        
        Args:
            epoch: Current epoch number
            train_dataloader: Training data loader
            optimizer: Optimizer for model update
            print_freq: Frequency of printing training status
            train_iters: Number of iterations per epoch
        """
        self.model.train()

        # Initialize metrics for monitoring training progress
        batch_time = AverageMeter()  # Time for processing each batch
        losses_gce = AverageMeter()  # Global classification loss
        losses_tri = AverageMeter()  # Triplet loss
        losses_cam = AverageMeter()  # Camera-aware loss
        losses_pce = AverageMeter()  # Part classification loss
        losses_diffusion = AverageMeter()  # Diffusion model loss
        losses_noise = AverageMeter()  # Noise prediction loss
        losses_consistency = AverageMeter()  # Consistency loss
        losses_contrastive = AverageMeter()  # Contrastive loss
        precisions = AverageMeter()  # Classification accuracy

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            # Load and parse data
            data = train_dataloader.next()
            inputs, targets, cams, ca = self._parse_data(data)

            # Forward pass through the model
            emb_g, emb_p, logits_g, logits_p, noise_pred, noisy_theta, noise = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # Compute various losses
            loss_gce = self.criterion_psclr(logits_g, logits_p, targets, ca)  # Global classification loss
            loss_tri = self.criterion_tri(emb_g, targets)  # Triplet loss
            loss_gcam = self.memory(emb_g, targets, cams)  # Global camera-aware loss

            # Compute part-level losses
            loss_pce = 0.
            loss_pcam = 0.
            if self.num_part > 0:
                if epoch >= self.se:
                    # Use label smoothing after starting epoch
                    for part in range(self.num_part):
                        loss_pce += self.criterion_pscls(logits_p[:, :, part], targets, ca[:, part])
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                else:
                    # Use standard cross entropy before starting epoch
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(logits_p[:, :, part], targets)
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                loss_pce /= self.num_part
                loss_pcam /= self.num_part

            # Compute diffusion model loss
            loss_diffusion, loss_dict = self.criterion_diffusion(
                noisy_theta,  # Predicted transformation parameters
                self.model.module.patch_proposal(self.model.module.base(inputs)),  # Initial transformation parameters
                noise_pred,  # Predicted noise
                noise,  # Ground truth noise
                emb_p,  # Part features for contrastive learning
                targets  # Person IDs for contrastive learning
            )

            # Combine all losses
            loss_cam = loss_pcam + loss_gcam
            loss = loss_gce + loss_pce + loss_tri + loss_cam * self.Wcam + loss_diffusion * self.Wdiff

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record statistics
            prec, = accuracy(logits_g.data, targets.data)

            # Update metrics
            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_cam.update(loss_cam.item())
            losses_pce.update(loss_pce.item())
            losses_diffusion.update(loss_diffusion.item())
            losses_noise.update(loss_dict['noise_loss'])
            losses_consistency.update(loss_dict['consistency_loss'])
            losses_contrastive.update(loss_dict['contrastive_loss'])
            precisions.update(prec[0])

            # Update time statistics
            batch_time.update(time.time() - end)
            end = time.time()

            # Print training status
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_CAM {:.3f} ({:.3f})\t'
                      'L_DIFF {:.3f} ({:.3f})\t'
                      'L_NOISE {:.3f} ({:.3f})\t'
                      'L_CONS {:.3f} ({:.3f})\t'
                      'L_CONT {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_cam.val, losses_cam.avg,
                              losses_diffusion.val, losses_diffusion.avg,
                              losses_noise.val, losses_noise.avg,
                              losses_consistency.val, losses_consistency.avg,
                              losses_contrastive.val, losses_contrastive.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        """Parse input data from data loader
        
        Args:
            inputs: Input data tuple containing:
                - imgs: Input images
                - _: Unused data
                - pids: Person IDs
                - cids: Camera IDs
                - idxs: Sample indices
            
        Returns:
            imgs: Input images on GPU
            pids: Person IDs on GPU
            cids: Camera IDs on GPU
            ca: Semantic consistency scores on GPU
        """
        imgs, _, pids, cids, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), cids.cuda(), ca.cuda()

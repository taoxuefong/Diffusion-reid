from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pisl import datasets
from pisl.loss import DiffusionThetaLoss
import maximum_mean_discrepancy
from pisl.models import resnet50part
from pisl.loss import CameraContrast
from pisl.trainers import PISLTrainerCAM
from pisl.evaluators import Evaluator, extract_all_features
from pisl.utils.data import IterLoader
from pisl.utils.data import transforms as T
from pisl.utils.data.sampler import RandomMultipleGallerySampler
from pisl.utils.data.preprocessor import Preprocessor
from pisl.utils.logging import Logger
from pisl.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance
from pisl.utils.serialization import load_checkpoint, copy_state_dict

best_mAP = 0


def get_data(name, data_dir):
    """Get dataset
    
    Args:
        name: Dataset name
        data_dir: Data directory
        
    Returns:
        dataset: Dataset object containing train, query and gallery sets
    """
    return datasets.create(name, data_dir)


def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, trainset=None):
    """Get training data loader with data augmentation
    
    Args:
        dataset: Dataset object
        height: Image height
        width: Image width
        batch_size: Batch size
        workers: Number of data loading workers
        num_instances: Number of instances per identity
        iters: Number of iterations
        trainset: Training set, if None use dataset.train
        
    Returns:
        train_loader: Training data loader with data augmentation
    """
    # Define data preprocessing and augmentation pipeline
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),  # Random horizontal flip for data augmentation
        T.Pad(10),  # Add padding for random crop
        T.RandomCrop((height, width)),  # Random crop for data augmentation
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])  # Random erasing for regularization
    ])

    # Get training set
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    
    # Create sampler for balanced sampling
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySampler(train_set, num_instances) if rmgs_flag else None
    
    # Create data loader with specified parameters
    train_loader = IterLoader(
        DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=not rmgs_flag,
            pin_memory=True,
            drop_last=True
        ),
        length=iters
    )

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    """Get test data loader without data augmentation
    
    Args:
        dataset: Dataset object
        height: Image height
        width: Image width
        batch_size: Batch size
        workers: Number of data loading workers
        testset: Test set, if None use dataset.query and dataset.gallery
        
    Returns:
        test_loader: Test data loader
    """
    # Define test data preprocessing (no augmentation)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    # Use query and gallery sets if testset not specified
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    """Compute pseudo labels using DBSCAN clustering
    
    Args:
        features: Feature vectors [N, D]
        cluster: DBSCAN clustering model
        k1: Number of neighbors for Jaccard distance computation
        
    Returns:
        labels: Pseudo labels for each sample
        num_ids: Number of unique clusters
    """
    # Compute Jaccard distance matrix
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    # Perform clustering
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    # Assign labels to samples
    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            # Assign unique labels to outliers
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


def compute_semantic_consistency(features_g, features_p, k, search_option=0):
    """Compute semantic consistency score between global and local features
    
    Args:
        features_g: Global features [N, D]
        features_p: Local features [N, D, P]
        k: Number of neighbors for consistency computation
        search_option: Search option for nearest neighbor computation
        
    Returns:
        consistency_scores: Semantic consistency score [N, P]
    """
    print("Compute semantic consistency score...")
    N, D, P = features_p.size()
    score = torch.zeros(N, P, device=features_g.device)
    end = time.time()
    
    # Compute ranking list for global features
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)
    
    # Pre-compute k-nearest neighbors for global features
    gb_neighbors_all = torch.stack([features_g[ranked_list_g[j]] for j in range(N)])  # shape: [N, k, D]
    
    # Process in batches to reduce memory usage
    initial_batch_size = 32
    min_batch_size = 16
    
    # Process in batches to reduce memory usage
    for i in range(P):
        # Compute ranking list for local features
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        
        # Process samples in batches
        for batch_start in range(0, N, initial_batch_size):
            batch_end = min(batch_start + initial_batch_size, N)
            batch_size_actual = batch_end - batch_start
            
            # Get k-nearest neighbors for local features in current batch
            pt_neighbors_batch = torch.stack([
                features_p[:, :, i][ranked_list_p_i[j]] for j in range(batch_start, batch_end)
            ])  # shape: [batch_size, k, D]
            
            # Get k-nearest neighbors for global features in current batch
            gb_neighbors_batch = gb_neighbors_all[batch_start:batch_end]  # shape: [batch_size, k, D]
            
            # Compute distance matrices in batch
            for j in range(batch_size_actual):
                # Compute distance matrices
                gb_dist = torch.cdist(gb_neighbors_batch[j], gb_neighbors_batch[j])  # shape: [k, k]
                pt_dist = torch.cdist(pt_neighbors_batch[j], pt_neighbors_batch[j])  # shape: [k, k]
                
                # Add distance normalization
                gb_dist = gb_dist / (gb_dist.max() + 1e-8)
                pt_dist = pt_dist / (pt_dist.max() + 1e-8)
                
                # Flatten distance matrices to 1D vectors
                gb_dist_flat = gb_dist.view(-1)  # shape: [k*k]
                pt_dist_flat = pt_dist.view(-1)  # shape: [k*k]
                
                # Compute MMD score
                mmd_score = maximum_mean_discrepancy.mmd_loss(
                    gb_dist_flat.unsqueeze(0),  # shape: [1, k*k]
                    pt_dist_flat.unsqueeze(0)   # shape: [1, k*k]
                ) / (k * 2)
                
                # Add stability control
                mmd_score = torch.clamp(mmd_score, min=0.0, max=1.0)
                
                score[batch_start + j, i] = mmd_score
    
    print("semantic consistency score time cost: {}".format(time.time() - end))
    
    return score


def main():
    """Main function for training and evaluation"""
    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def main_worker(args):
    """Main worker function for training and evaluation
    
    Args:
        args: Command line arguments
    """
    global best_mAP

    cudnn.benchmark = True

    # Setup logging
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Load dataset
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                     testset=sorted(dataset.train))
                                     
    # Initialize model
    num_parts = args.part
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)
    
    evaluator = Evaluator(model)

    # Setup optimizer and learning rate scheduler
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Training loop
    consistency_score_log = torch.FloatTensor([])
    for epoch in range(args.epochs):
        # Extract features for clustering
        global_features, part_features, _ = extract_all_features(model, cluster_loader)
        global_features = torch.cat([global_features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        part_features = torch.cat([part_features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

        # Initialize DBSCAN clustering
        if epoch == 0:
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)

        # Assign pseudo-labels using clustering
        pseudo_labels, num_classes = compute_pseudo_labels(global_features, cluster, args.k1)

        # Compute semantic consistency scores
        consistency_scores = compute_semantic_consistency(global_features, part_features, k=args.knn)
        consistency_score_log = torch.cat([consistency_score_log, consistency_scores.unsqueeze(0)], dim=0)

        # Generate new dataset with pseudo-labels
        num_outliers = 0
        new_dataset = []

        sample_indices, camera_ids, person_ids = [], [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pid = label.item()
            if pid >= num_classes:  # Skip outliers
                num_outliers += 1
            else:
                new_dataset.append((fname, pid, cid))
                sample_indices.append(i)
                camera_ids.append(cid)
                person_ids.append(pid)

        # Create training data loader with new dataset
        train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=new_dataset)

        # Print clustering statistics
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_classes,
                                                                                           num_outliers))

        # Reindex features and scores
        sample_indices = np.asarray(sample_indices)
        camera_ids = np.asarray(camera_ids)
        person_ids = np.asarray(person_ids)
        global_features = global_features[sample_indices, :]
        part_features = part_features[sample_indices, :, :]
        consistency_scores = consistency_scores[sample_indices, :]

        # Compute cluster centroids and camera-aware proxies
        global_centroids, part_centroids = [], []
        camera_proxies, camera_part_proxies, proxy_pids, proxy_cids = [], [], [], []
        for pid in sorted(np.unique(person_ids)):  # loop all pids
            pid_indices = np.where(person_ids == pid)[0]
            global_centroids.append(global_features[pid_indices].mean(0))
            part_centroids.append(part_features[pid_indices].mean(0))

            for cid in sorted(np.unique(camera_ids[pid_indices])):  # loop all cids for pid
                cid_indices = np.where(camera_ids == cid)[0]
                common_indices = np.intersect1d(pid_indices, cid_indices)
                camera_proxies.append(global_features[common_indices].mean(0))
                camera_part_proxies.append(part_features[common_indices].mean(0))
                proxy_pids.append(pid)
                proxy_cids.append(cid)

        # Initialize global feature memory
        global_centroids = F.normalize(torch.stack(global_centroids), p=2, dim=1)
        model.module.classifier.weight.data[:num_classes].copy_(global_centroids)
        camera_memory = CameraContrast(global_centroids.size(1), len(proxy_pids)).cuda()
        camera_memory.proxy = F.normalize(torch.stack(camera_proxies), p=2, dim=1).cuda()
        camera_memory.pids = torch.Tensor(proxy_pids).long().cuda()
        camera_memory.cids = torch.Tensor(proxy_cids).long().cuda()

        # Initialize part feature memories
        part_memories = []
        for i in range(num_parts):
            part_centroids_i = torch.stack(part_centroids)[:, :, i]
            part_centroids_i = F.normalize(part_centroids_i, p=2, dim=1)
            part_classifier = getattr(model.module, 'classifier' + str(i))
            part_classifier.weight.data[:num_classes].copy_(part_centroids_i)

            part_memory = CameraContrast(global_centroids.size(1), len(proxy_pids)).cuda()
            camera_part_proxies_i = torch.stack(camera_part_proxies)[:, :, i]
            part_memory.proxy = F.normalize(camera_part_proxies_i, p=2, dim=1).cuda()
            part_memory.pids = torch.Tensor(proxy_pids).long().cuda()
            part_memory.cids = torch.Tensor(proxy_cids).long().cuda()
            part_memories.append(part_memory)

        # Training
        trainer = PISLTrainerCAM(model, consistency_scores, camera_memory, part_memories, 
                               num_class=num_classes, num_part=num_parts,
                               Wref=args.Wref, se=args.se, Wcam=args.Wcam, Wdiff=args.Wdiff)

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
        lr_scheduler.step()

        # Evaluation
        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    # Save final model and consistency scores
    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), consistency_score_log.numpy())

    # Load best model and evaluate
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unsupervised Learning of Intrinsic Semantics With Diffusion Model for Person Re-Identification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)

    # PISL
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--knn', type=int, default=20,
                        help="hyperparameter for semantic consistency score")
    parser.add_argument('--Wref', type=float, default=0.5,
                        help="weighting parameter for label refinement")
    parser.add_argument('--se', type=int, default=5,
                        help="starting epoch for label smoothing")
    parser.add_argument('--Wcam', type=float, default=0.5,
                        help="weighting parameter of camera contrastive loss")
    parser.add_argument('--Wdiff', type=float, default=0.1,
                        help="weighting parameter of diffusion loss")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # cluster
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")

    main()

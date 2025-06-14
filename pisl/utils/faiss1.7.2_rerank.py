#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu, get_gpu_resources

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1].long()
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    res = get_gpu_resources()  # 使用新的GPU资源配置

    # 将特征分批处理以减少内存使用
    N = target_features.size(0)
    batch_size = 512  # 减小批处理大小以降低内存使用
    initial_rank = torch.zeros(N, k1, device=target_features.device, dtype=torch.long)  # 确保是 long 类型
    
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch_features = target_features[i:end_idx]
        # 对每个批次进行k近邻搜索
        _, batch_initial_rank = search_raw_array_pytorch(res, target_features, batch_features, k1)
        initial_rank[i:end_idx] = batch_initial_rank.long()  # 确保是 long 类型
        
        if print_flag and i % (5*batch_size) == 0:
            print(f'Processing {i}/{N} ...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=np.float16 if use_float16 else np.float32)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(np.float16 if use_float16 else np.float32)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=np.float16 if use_float16 else np.float32)
    for i in range(N):
        temp_min = np.zeros((1,N), dtype=np.float16 if use_float16 else np.float32)
        # temp_max = np.zeros((1,N), dtype=np.float16 if use_float16 else np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist

@torch.no_grad()
def compute_ranked_list(features, k=20, search_option=0, fp16=False, verbose=True):

    end = time.time()
    if verbose:
        print("Computing ranked list...")

    if search_option < 3:
        torch.cuda.empty_cache()
        features = features.cuda().detach()

    ngpus = faiss.get_num_gpus()

    if search_option == 0:
        # Faiss Search + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k+1)
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 1:
        # Faiss Search + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k+1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()

    elif search_option == 2:
        # PyTorch Search + PyTorch CUDA Tensors
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=True)
        initial_rank = torch.argsort(dist_m, dim=1)
        initial_rank = initial_rank.cpu().numpy()

    else:
        # Numpy Search (CPU)
        torch.cuda.empty_cache()
        features = features.cuda().detach()
        dist_m = compute_euclidean_distance(features, cuda=False)
        initial_rank = np.argsort(dist_m.cpu().numpy(), axis=1)
        features = features.cpu()

    features = features.cpu()
    if verbose:
        print("Ranked list computing time cost: {}".format(time.time() - end))

    return initial_rank[:, 1:k+1]

@torch.no_grad()
def compute_euclidean_distance(features, others=None, cuda=False):
    if others is None:
        if cuda:
            features = features.cuda()

        n = features.size(0)
        x = features.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        del features

    else:
        if cuda:
            features = features.cuda()
            others = others.cuda()

        m, n = features.size(0), others.size(0)
        dist_m = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n) +\
                 torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist_m.addmm_(features, others.t(), beta=1, alpha=-2)
        del features, others

    return dist_m

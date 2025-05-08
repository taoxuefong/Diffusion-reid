import os
import numpy as np
import faiss
import torch

def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)

def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)

def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    
    # 将输入转换为numpy数组
    x_np = x.cpu().numpy()
    
    # 使用FAISS搜索
    D_np, I_np = index.search(x_np, k)
    
    # 将结果转回PyTorch张量
    D.copy_(torch.from_numpy(D_np))
    I.copy_(torch.from_numpy(I_np))
    
    torch.cuda.synchronize()
    return D, I

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(512 * 1024 * 1024)  # 设置临时内存为512MB
    return res

def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    """使用FAISS GPU进行k近邻搜索"""
    assert xb.device == xq.device
    
    # 获取维度信息
    nq, d = xq.size()
    nb, d2 = xb.size()
    assert d2 == d
    
    # 初始化输出张量
    if D is None:
        D = torch.empty((nq, k), dtype=torch.float32, device=xb.device)
    if I is None:
        I = torch.empty((nq, k), dtype=torch.int64, device=xb.device)
        
    # 创建GPU索引
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = torch.cuda.current_device()
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    
    # 添加基准点
    index.add(xb.cpu().numpy())
    
    # 搜索
    D_np, I_np = index.search(xq.cpu().numpy(), k)
    
    # 将结果复制回GPU
    D.copy_(torch.from_numpy(D_np))
    I.copy_(torch.from_numpy(I_np))
    
    return D, I

def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index

def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)

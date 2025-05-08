import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from pisl.models import build_model
from pisl.utils.data import get_test_loader
from pisl.utils.metrics import extract_features, compute_rank_list


def visualize_ranking(query_img, gallery_imgs, query_label, gallery_labels, 
                     save_path, top_k=10):
    """Visualize ranking results
    
    Args:
        query_img: Query image tensor [C x H x W]
        gallery_imgs: Gallery image tensors [N x C x H x W]
        query_label: Query image label
        gallery_labels: Gallery image labels [N]
        save_path: Path to save visualization
        top_k: Number of top results to show
    """
    # Convert tensors to numpy arrays
    query_img = query_img.cpu().numpy().transpose(1, 2, 0)
    gallery_imgs = gallery_imgs.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    query_img = std * query_img + mean
    gallery_imgs = std * gallery_imgs + mean
    
    # Clip values to [0, 1]
    query_img = np.clip(query_img, 0, 1)
    gallery_imgs = np.clip(gallery_imgs, 0, 1)
    
    # Create visualization
    plt.figure(figsize=(15, 3))
    
    # Plot query image
    plt.subplot(1, top_k + 1, 1)
    plt.imshow(query_img)
    plt.title(f'Query\nID: {query_label}')
    plt.axis('off')
    
    # Plot gallery images
    for i in range(top_k):
        plt.subplot(1, top_k + 1, i + 2)
        plt.imshow(gallery_imgs[i])
        plt.title(f'Rank {i+1}\nID: {gallery_labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Configuration
    model_path = 'path/to/your/model.pth'  # 替换为您的模型路径
    dataset_name = 'market1501'  # 替换为您的数据集名称
    save_dir = 'visualization/ranking'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = build_model(name='resnet50', num_classes=751, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    
    # Get test loader
    test_loader = get_test_loader(dataset_name, batch_size=32, num_workers=4)
    
    # Extract features
    features, labels = extract_features(model, test_loader)
    
    # Compute ranking list
    rank_list = compute_rank_list(features, labels)
    
    # Visualize ranking results for a few queries
    num_queries = 5
    for i in range(num_queries):
        query_idx = i
        query_label = labels[query_idx]
        
        # Get top-k gallery images
        top_k = 10
        gallery_indices = rank_list[query_idx][:top_k]
        gallery_labels = labels[gallery_indices]
        
        # Get images
        query_img = test_loader.dataset[query_idx][0]
        gallery_imgs = torch.stack([test_loader.dataset[idx][0] for idx in gallery_indices])
        
        # Visualize ranking
        save_path = os.path.join(save_dir, f'ranking_query_{query_idx}.png')
        visualize_ranking(query_img, gallery_imgs, query_label, gallery_labels, save_path)


if __name__ == '__main__':
    main() 
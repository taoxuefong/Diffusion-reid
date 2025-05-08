import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from pisl.models import build_model
from pisl.utils.data import get_test_loader
from pisl.utils.metrics import extract_features


def visualize_tsne(features, labels, save_path, perplexity=30, n_iter=1000):
    """Visualize features using t-SNE
    
    Args:
        features: Feature vectors [N x D]
        labels: Class labels [N]
        save_path: Path to save visualization
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
    features_2d = tsne.fit_transform(features)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each class with different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=f'ID {label}', alpha=0.6)
    
    plt.title('t-SNE Visualization of Features')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Configuration
    model_path = 'path/to/your/model.pth'  # 替换为您的模型路径
    dataset_name = 'market1501'  # 替换为您的数据集名称
    save_dir = 'visualization/tsne'
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
    
    # Convert to numpy arrays
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Visualize t-SNE
    save_path = os.path.join(save_dir, 'tsne_visualization.png')
    visualize_tsne(features, labels, save_path)


if __name__ == '__main__':
    main() 
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from pisl.models import build_model
from pisl.utils.data import get_test_loader
from pisl.utils.metrics import extract_features


def visualize_heatmap(model, img_path, save_path, transform=None):
    """Visualize attention heatmap for a single image
    
    Args:
        model: PISL model
        img_path: Path to input image
        save_path: Path to save visualization
        transform: Image transformation
    """
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    if transform is None:
        transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    
    # Get model attention
    model.eval()
    with torch.no_grad():
        # Get attention maps from the model
        attention_maps = model.get_attention_maps(img_tensor)
        
        # Convert attention maps to numpy arrays
        attention_maps = attention_maps.cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention heatmap
        plt.subplot(1, 3, 2)
        heatmap = np.mean(attention_maps[0], axis=0)  # Average over channels
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        plt.imshow(heatmap, cmap='jet')
        plt.title('Attention Heatmap')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def main():
    # Configuration
    model_path = 'path/to/your/model.pth'  # 替换为您的模型路径
    dataset_name = 'market1501'  # 替换为您的数据集名称
    save_dir = 'visualization/heatmaps'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = build_model(name='resnet50', num_classes=751, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    
    # Get test loader
    test_loader = get_test_loader(dataset_name, batch_size=1, num_workers=4)
    
    # Visualize heatmaps for a few samples
    for i, (imgs, _, _, _, _) in enumerate(tqdm(test_loader)):
        if i >= 10:  # 只可视化前10张图片
            break
            
        img_path = test_loader.dataset.imgs[i]
        save_path = os.path.join(save_dir, f'heatmap_{i}.png')
        visualize_heatmap(model, img_path, save_path)


if __name__ == '__main__':
    main() 
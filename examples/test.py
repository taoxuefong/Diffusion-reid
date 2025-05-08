from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from pisl import datasets
from pisl.models import resnet50part
from pisl.evaluators import Evaluator
from pisl.utils.data import transforms as T
from pisl.utils.data.preprocessor import Preprocessor
from pisl.utils.logging import Logger
from pisl.utils.serialization import load_checkpoint, copy_state_dict


def get_data(name, data_dir, height, width, batch_size, workers):
    """Get dataset and data loader
    
    Args:
        name: Dataset name
        data_dir: Data directory
        height: Image height
        width: Image width
        batch_size: Batch size
        workers: Number of data loading workers
        
    Returns:
        dataset: Dataset object
        test_loader: Test data loader
    """
    dataset = datasets.create(name, data_dir)
    
    # Define data preprocessing
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    
    # Merge query and gallery datasets
    test_set = list(set(dataset.query) | set(dataset.gallery))
    
    # Create data loader
    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True
    )

    return dataset, test_loader


def main():
    args = parser.parse_args()

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
    """Main worker function
    
    Args:
        args: Command line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Load dataset
    dataset, test_loader = get_data(
        args.dataset, 
        args.data_dir, 
        args.height, 
        args.width, 
        args.batch_size, 
        args.workers
    )

    # Create and load model
    model = create_model(args.part)
    load_checkpoint_to_model(args.resume, model)

    # Evaluate model
    evaluator = Evaluator(model)
    print(f"Test on {args.dataset}:")
    evaluator.evaluate(
        test_loader, 
        dataset.query, 
        dataset.gallery, 
        cmc_flag=True, 
        rerank=args.rerank
    )

def set_seed(seed):
    """Set random seed
    
    Args:
        seed: Random seed
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def create_model(num_parts):
    """Create model
    
    Args:
        num_parts: Number of parts
        
    Returns:
        model: Model object
    """
    model = resnet50part(num_parts=num_parts, num_classes=3000)
    model.cuda()
    return nn.DataParallel(model)

def load_checkpoint_to_model(checkpoint_path, model):
    """Load checkpoint to model
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model object
    """
    checkpoint = load_checkpoint(checkpoint_path)
    copy_state_dict(checkpoint, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))

    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true', help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)

    # model configs
    parser.add_argument('--part', type=int, default=3, help="number of part")

    main()

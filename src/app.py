"""
Main application for cell segmentation using Attention U-Net.

This script provides testing functionality for the trained Attention U-Net model.
"""

import os
import torch
import yaml
import argparse
from torch.utils.data import DataLoader

# Import local modules
from src.model import AttentionUNet, device
from src.data import CellDataset
from src.inference import prepare_inference_directory, run_inference


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dict with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path):
    """
    Load trained model from file.
    
    Args:
        model_path: Path to model state dict
        
    Returns:
        Loaded model
    """
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def get_image_paths(img_dirs):
    """
    Get all image paths from specified directories.
    
    Args:
        img_dirs: List of image directory paths
        
    Returns:
        List of absolute paths to images
    """
    image_paths = []
    
    # Traverse each directory and collect image paths
    for img_dir in img_dirs:
        if os.path.exists(img_dir):
            image_paths.extend([
                os.path.join(img_dir, img) 
                for img in os.listdir(img_dir) 
                if os.path.isfile(os.path.join(img_dir, img))
            ])
        else:
            print(f"Directory {img_dir} does not exist")
    
    return image_paths


def main(config_path):
    """
    Main entry point for the application.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Load model
    model = load_model(config['model_path'])
    
    # Get image paths
    image_paths = get_image_paths(config['test_img_dirs'])
    
    # Create dataset and dataloader
    test_dataset = CellDataset(image_paths, None, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers']
    )
    
    # Prepare output directory
    output_dir = prepare_inference_directory(config.get('output_dir'))
    
    # Run inference
    output_paths = run_inference(
        model, 
        test_loader, 
        device, 
        output_dir, 
        threshold=config['threshold']
    )
    
    print(f"Inference completed. Results saved to {output_dir}")
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Segmentation Inference")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/test_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    main(args.config)

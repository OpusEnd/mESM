"""
Main entry point for UNetSLSM that matches the original script functionality.

This script mimics the functionality of the original test.py script but uses the modular structure.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the parent directory to the sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our modular package
from src.model import AttentionUNet, device
from src.data import CellDataset
from src.inference import prepare_inference_directory, run_inference


def main(config_path="config/test_config.yaml", force_cpu=False):
    """Run inference using settings similar to the original script."""
    # Print some info about the environment
    global device  # Use the global device variable
    
    # Override device if force_cpu is True
    if force_cpu:
        import torch
        device = torch.device('cpu')
        print("Forcing CPU usage as requested")
    
    print(f"Using device: {device}")
    print(f"Current directory: {os.getcwd()}")
      # Load configuration
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
    else:
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        config = {
            "model_path": "runs/light_unet_training_20241024_175658/final_model.pth",
            "threshold": 0.07,
            "test_img_dirs": ["data/cell_dataset/img_dir/Stest"],
            "batch_size": 2,
            "num_workers": 1
        }
      # Load model if available
    model_path = config.get("model_path", "runs/light_unet_training_20241024_175658/final_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Creating a new model instance without loading weights (for testing only)")
        model = AttentionUNet().to(device)
    else:
        print(f"Loading model from {model_path}")
        model = AttentionUNet().to(device)
        try:
            # Load the state dict with appropriate device handling
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instance without loading weights")
    
    model.eval()
      # Get test image paths
    test_img_dirs = config.get("test_img_dirs", ["data/cell_dataset/img_dir/Stest"])
    image_paths = []
    
    for test_dir in test_img_dirs:
        if os.path.exists(test_dir):
            image_paths.extend([
                os.path.join(test_dir, img) 
                for img in os.listdir(test_dir) 
                if os.path.isfile(os.path.join(test_dir, img))
            ])
        else:
            print(f"Warning: Test directory {test_dir} not found")
    
    if not image_paths:
        print(f"Warning: No images found in {test_dir}")
        print("Creating a dummy test image for demonstration")
        import numpy as np
        from PIL import Image
        # Create a dummy test image
        dummy_img = np.ones((128, 128), dtype=np.uint8) * 128
        dummy_path = os.path.join(test_dir, "dummy_test.png")
        Image.fromarray(dummy_img).save(dummy_path)
        image_paths = [dummy_path]
      # Create dataset and dataloader
    print(f"Found {len(image_paths)} test images")
    test_dataset = CellDataset(image_paths, None, mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.get("batch_size", 2),
        shuffle=False, 
        num_workers=config.get("num_workers", 1)
    )
    
    # Create output directory
    output_dir = prepare_inference_directory(config.get("output_dir", None))
    print(f"Results will be saved to {output_dir}")
    
    # Run inference
    try:
        output_paths = run_inference(
            model, 
            test_loader,
            device, 
            output_dir, 
            threshold=config.get("threshold", 0.07)
        )
        print(f"Generated {len(output_paths)} output files")
    except Exception as e:
        print(f"Error during inference: {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()

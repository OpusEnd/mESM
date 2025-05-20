"""
Inference module for cell segmentation using Attention U-Net model.

This module provides functions for making predictions on new images.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from datetime import datetime


def prepare_inference_directory(output_dir=None):
    """
    Prepare directory for saving inference results.
    
    Args:
        output_dir: Custom output directory path (optional)
        
    Returns:
        Path to the created directory
    """
    if output_dir is None:
        output_dir = f'outputs/test_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_inference(model, test_loader, device, output_dir, threshold=0.07):
    """
    Run inference on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test images
        device: Computation device (CPU or GPU)
        output_dir: Directory to save results
        threshold: Threshold value for binary prediction (default: 0.07)
        
    Returns:
        List of output image paths
    """
    output_paths = []
    
    # Make sure the model is in evaluation mode
    model.eval()
    
    # Run inference with no gradient tracking
    with torch.no_grad():
        for idx, imgs in enumerate(test_loader):
            # Move images to device and get predictions
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = (outputs > threshold).float().cpu().numpy()

            # Process predictions for each image in the batch
            for i in range(imgs.shape[0]):
                # Create binary mask
                pred_mask = (preds[i, 0] * 255).astype(np.uint8)

                # Convert original image for visualization
                original_image = imgs[i, 0].cpu().numpy() * 255
                original_image = np.clip(original_image, 0, 255).astype(np.uint8)

                # Create a green mask (using the green channel)
                color_mask = np.zeros_like(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR))
                color_mask[:, :, 0] = 0  # No blue
                color_mask[:, :, 1] = pred_mask  # Green channel for the mask
                color_mask[:, :, 2] = 0  # No red

                # Blend original image and colored mask
                overlay = cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR), 0.2, 
                                         color_mask, 0.8, 0)

                # Save overlay image
                overlay_path = os.path.join(output_dir, f'test_overlay_{idx * imgs.shape[0] + i}.png')
                print(f"Saving overlay to: {overlay_path}")
                overlay_image = Image.fromarray(overlay)
                overlay_image.save(overlay_path)
                output_paths.append(overlay_path)

                # Save binary mask
                mask_path = os.path.join(output_dir, f'test_mask_{idx * imgs.shape[0] + i}.png')
                print(f"Saving mask to: {mask_path}")
                mask_image = Image.fromarray(pred_mask)
                mask_image.save(mask_path)
                output_paths.append(mask_path)
                
    return output_paths

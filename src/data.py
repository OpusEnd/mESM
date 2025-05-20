"""
Data processing module for cell image segmentation.

This module provides functions for loading, processing, and augmenting cell images.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter


def size_padding(img, target_size=(3008, 4112)):
    """
    Pad image to target size.
    
    Args:
        img: Input image
        target_size: Desired output size (height, width)
        
    Returns:
        Padded image
    """
    current_height, current_width = img.shape[:2]
    target_height, target_width = target_size
    
    # Calculate padding values
    delta_height = target_height - current_height
    delta_width = target_width - current_width

    if delta_height <= 0 or delta_width <= 0:
        return img
    
    top = 0
    bottom = delta_height
    left = 0
    right = delta_width 

    # Apply padding
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    return padded_img


def adjust_contrast_and_save(img, save_path):
    """
    Adjust contrast of image using morphological top-hat transform and save it.
    
    Args:
        img: Input image (grayscale)
        save_path: Path to save the adjusted image
        
    Returns:
        Contrast-adjusted image (float32 in range [0.2, 0.8])
    """
    try:
        # Ensure image is in correct format for processing
        if img is None:
            print(f"Warning: Input image is None in adjust_contrast_and_save")
            img = np.ones((128, 128), dtype=np.float32) * 0.5
            
        # Ensure image is float32 and in [0,1] range
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0 if img.max() <= 255 else img / 65535.0
            
        # Step 1: Apply morphological top-hat transform
        kernel_size = min(10, max(3, min(img.shape) // 10))  # Adaptive kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        
        # Step 2: Use percentiles to determine range, reducing impact of extreme values
        lower_percentile = np.percentile(tophat, 1)
        upper_percentile = np.percentile(tophat, 99)
        
        # Step 3: Clip image to determined range
        img_clipped = np.clip(tophat, lower_percentile, upper_percentile)
        
        # Step 4: Scale pixel values to [0.2, 0.8] range
        img_min, img_max = img_clipped.min(), img_clipped.max()
        if img_min == img_max:  # Handle constant value image
            scaled_img = np.ones_like(img_clipped) * 0.5
        else:
            scaled_img = (img_clipped - img_min) / (img_max - img_min)  # Normalize to [0, 1]
            scaled_img = 0.2 + scaled_img * (0.8 - 0.2)  # Scale to [0.2, 0.8]
        
        # Step 5: Convert to 8-bit (0-255)
        img_8bit = (scaled_img * 255).astype('uint8')
        
        # Step 6: Save as 8-bit TIFF using PIL
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_8bit_pil = Image.fromarray(img_8bit)
        img_8bit_pil.save(save_path, format='TIFF')
        
        return scaled_img  # Return adjusted image
        
    except Exception as e:
        print(f"Error in adjust_contrast_and_save: {str(e)}")
        # Return a placeholder image on error
        return np.ones_like(img) * 0.5


def generate_class_weight_map(label, foreground_weight=2, background_weight=1):
    """
    Generate a weight map based on class imbalance.
    
    Args:
        label: Binary label image (1 for foreground, 0 for background)
        foreground_weight: Weight for foreground pixels
        background_weight: Weight for background pixels
        
    Returns:
        Weight map (same shape as label)
    """
    weight_map = np.zeros_like(label, dtype=np.float32)
    weight_map[label == 1] = foreground_weight
    weight_map[label == 0] = background_weight
    return weight_map


class CellDataset(Dataset):
    """
    Dataset class for cell segmentation.
    """
    def __init__(self, image_paths, label_paths, transform=None, mode='train'):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to input images
            label_paths: List of paths to label images (None for test mode)
            transform: Albumentations transforms to apply
            mode: 'train', 'val', or 'test'
        """
        self.image_paths = image_paths
        self.label_paths = label_paths if mode != 'test' else None
        self.mode = mode
        self.transform = transform

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            For train/val: tuple of (image, label, weight_map)
            For test: image tensor
        """
        img_path = self.image_paths[idx]        # Read image with improved error handling and format detection
        try:
            # First try reading as-is (supports most formats)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # If can't read or if image is PGM (for test image)
            if img is None or img_path.lower().endswith('.pgm'):
                # Try alternative method for PGM or other formats
                from PIL import Image
                pil_img = Image.open(img_path)
                img = np.array(pil_img)
                
            # Still failed? Try one more time with different flag
            if img is None:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
            # Final check
            if img is None:
                raise FileNotFoundError(f"Image at path {img_path} could not be loaded with any method.")
                
            # Ensure correct normalization based on bit depth
            if img.dtype == np.uint16:
                img = (img / 65535.0).astype('float32')  # Normalize 16-bit to [0, 1]
            else:
                img = (img / 255.0).astype('float32')  # Normalize 8-bit to [0, 1]
                
            # Ensure image has correct dimensions (H,W)
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Create a small dummy image for testing
            img = np.ones((128, 128), dtype=np.float32) * 0.5# Save adjusted image (for record keeping purposes only)
        adjusted_img_dir = f'data/BF_dataset/adjusted_image/{self.mode}/'
        os.makedirs(adjusted_img_dir, exist_ok=True)
        save_path = os.path.join(adjusted_img_dir, os.path.basename(img_path).replace('.tif', '_adjusted.tiff'))
        if self.mode == 'test':
            img = adjust_contrast_and_save(size_padding(img), save_path)  # Get the adjusted image
        else:
            img = adjust_contrast_and_save(img, save_path)

        # Generate Otsu threshold map after contrast adjustment
        otsu_map = cv2.threshold((img * 255).astype('uint8'), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype('float32')
        
        # Generate distance transform map after contrast adjustment
        distance_transform = cv2.distanceTransform((otsu_map * 255).astype('uint8'), cv2.DIST_L2, 5)
        distance_transform = (distance_transform / distance_transform.max()).astype('float32')

        # Stack original image, Otsu map, and distance transform as input channels
        img_combined = np.stack([img, otsu_map, distance_transform], axis=0)  # Shape: (3, H, W)

        if self.mode != 'test':
            label_path = self.label_paths[idx]
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # Read label as grayscale
            if label is None:
                raise FileNotFoundError(f"Label at path {label_path} could not be loaded. Please check the file path.")
            label = (label / 255.0).astype('float32')  # Normalize to [0, 1] since label is binary

            # Generate foreground-background weight map
            weight_map = generate_class_weight_map(label)  # [H, W]
            weight_map = weight_map.astype('float32')

            if self.transform:
                augmented = self.transform(image=img_combined.transpose(1, 2, 0), mask=label, mask2=weight_map)
                img_combined = augmented['image'].float()  # Already converts to (C, H, W)
                weight_map = augmented['mask2']
                label = augmented['mask']

            # Convert numpy arrays to PyTorch tensors
            img_combined = torch.tensor(img_combined, dtype=torch.float32)
            weight_map = torch.tensor(weight_map)
            label = torch.tensor(label)

            # Check and ensure dimensions are correct: add channel dimension if needed
            if label.ndim == 2:
                label = label.unsqueeze(0)  # From (H, W) -> (1, H, W)
            if weight_map.ndim == 2:
                weight_map = weight_map.unsqueeze(0)  # From (H, W) -> (1, H, W)
            
            return img_combined, label, weight_map
            
        # If test mode, we only need the image
        if self.transform:
            augmented = self.transform(image=img_combined.transpose(1, 2, 0))
            img_combined = augmented['image'].permute(2, 0, 1).float()
        else:
            img_combined = torch.tensor(img_combined)

        return img_combined


def dynamic_crop(image, **kwargs):
    """
    Dynamically crop the image based on its size.
    
    Args:
        image: Input image
        kwargs: Additional keyword arguments
        
    Returns:
        Cropped image
    """
    # Check if the image has multiple channels
    if image.ndim == 3:
        height, width, channels = image.shape
        crop_height = min(height, 128)
        crop_width = min(width, 128)
        start_x = np.random.randint(0, width - crop_width + 1) if width > crop_width else 0
        start_y = np.random.randint(0, height - crop_height + 1) if height > crop_height else 0
        cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
    else:
        height, width = image.shape[:2]
        crop_height = min(height, 128)
        crop_width = min(width, 128)
        start_x = np.random.randint(0, width - crop_width + 1) if width > crop_width else 0
        start_y = np.random.randint(0, height - crop_height + 1) if height > crop_height else 0
        cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    
    return cropped_image

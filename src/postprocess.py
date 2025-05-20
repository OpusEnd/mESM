"""
Post-processing module for cell segmentation masks.

This module provides functions for cleaning up segmentation masks,
including smoothing, removing small regions, and filling holes.
"""

import cv2
import numpy as np
import os


def smooth_mask(mask, kernel_size=(1, 1), sigma=0):
    """
    Apply Gaussian smoothing to a binary mask.
    
    Args:
        mask: Binary mask to smooth
        kernel_size: Size of Gaussian kernel (default: (1, 1))
        sigma: Standard deviation for Gaussian kernel (default: 0)
        
    Returns:
        Smoothed binary mask
    """
    smoothed = cv2.GaussianBlur(mask, kernel_size, sigma)
    # Convert back to binary
    _, binary = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_regions(mask, min_area=3500):
    """
    Remove regions smaller than the specified area.
    
    Args:
        mask: Binary mask to process
        min_area: Minimum area to keep (in pixels) (default: 3500)
        
    Returns:
        Cleaned mask with small regions removed
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    return cleaned_mask


def fill_holes(mask):
    """
    Fill holes in the mask using flood fill from the outside.
    
    Args:
        mask: Binary mask to process
        
    Returns:
        Mask with holes filled
    """
    # Copy the mask
    filled_mask = mask.copy()
    
    # Create a mask for flood fill (needs to be 2 pixels larger)
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Flood fill from the (0,0) - outside of the shape
    cv2.floodFill(filled_mask, flood_mask, (0, 0), 255)
    
    # Invert and combine with original to fill holes
    filled_mask = cv2.bitwise_not(filled_mask)
    final_mask = cv2.bitwise_or(mask, filled_mask)
    
    return final_mask


def process_mask(mask_path, output_path=None, min_area=3500, 
                 kernel_size=(1, 1), sigma=0):
    """
    Process a mask image by smoothing, removing small regions, and filling holes.
    
    Args:
        mask_path: Path to the mask image
        output_path: Path to save the processed mask (default: None)
        min_area: Minimum area for regions to keep (default: 3500)
        kernel_size: Size of Gaussian kernel (default: (1, 1))
        sigma: Standard deviation for Gaussian kernel (default: 0)
        
    Returns:
        Processed mask as a numpy array
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask file: {mask_path}")
    
    # Ensure binary mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Process mask
    smoothed_mask = smooth_mask(mask, kernel_size, sigma)
    cleaned_mask = remove_small_regions(smoothed_mask, min_area)
    final_mask = fill_holes(cleaned_mask)
    
    # Save result if output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, final_mask)
        print(f"Processed mask saved to: {output_path}")
    
    return final_mask


def create_overlay(mask, original_image_path, output_path=None, alpha=0.3):
    """
    Create an overlay of the mask on the original image.
    
    Args:
        mask: Processed binary mask
        original_image_path: Path to the original image
        output_path: Path to save the overlay (default: None)
        alpha: Opacity of the mask overlay (default: 0.3)
        
    Returns:
        Overlay image as a numpy array
    """
    # Load original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        raise ValueError(f"Cannot read original image: {original_image_path}")
    
    # Convert grayscale to BGR if needed
    if len(original_image.shape) == 2:
        original_bgr = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_image
    
    # Resize mask if needed
    h_orig, w_orig = original_bgr.shape[:2]
    if mask.shape[:2] != (h_orig, w_orig):
        mask = mask[:h_orig, :w_orig]
    
    # Create a blue mask
    blue_mask = np.zeros_like(original_bgr)
    blue_mask[:, :, 0] = mask  # Blue channel
    
    # Create overlay
    overlay = cv2.addWeighted(original_bgr, 1-alpha, blue_mask, alpha, 0)
    
    # Save result if output path is provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Overlay saved to: {output_path}")
    
    return overlay


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process segmentation masks")
    parser.add_argument("--mask_path", type=str, required=True, 
                        help="Path to the mask image")
    parser.add_argument("--original_image", type=str, required=True,
                        help="Path to the original image")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the processed masks")
    parser.add_argument("--min_area", type=int, default=3500,
                        help="Minimum area for regions to keep")
    
    args = parser.parse_args()
    
    # Set output paths
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        processed_mask_path = os.path.join(args.output_dir, "processed_mask_final.png")
        overlay_path = os.path.join(args.output_dir, "mask_overlay.png")
    else:
        base_dir = os.path.dirname(args.mask_path)
        processed_mask_path = os.path.join(base_dir, "processed_mask_final.png")
        overlay_path = os.path.join(base_dir, "mask_overlay.png")
    
    # Process mask
    final_mask = process_mask(args.mask_path, processed_mask_path, args.min_area)
    
    # Create overlay
    create_overlay(final_mask, args.original_image, overlay_path)
    
    print("Mask processing completed.")
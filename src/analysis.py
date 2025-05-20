"""
Intensity extraction and analysis module for cell segmentation.

This module provides functions for extracting intensity statistics from cell regions,
including creation of ring masks, quadrant and octant region analysis.
"""

import cv2
import numpy as np
import os
import csv
from multiprocessing import Pool, cpu_count
from functools import partial

# Constants
DEFAULT_RING_WIDTH = 20
DEFAULT_CSV_HEADERS = [
    'Frame',
    'Original_Mean', 'Original_Std',
    'Original_Q1_Mean', 'Original_Q1_Std',
    'Original_Q2_Mean', 'Original_Q2_Std',
    'Original_Q3_Mean', 'Original_Q3_Std',
    'Original_Q4_Mean', 'Original_Q4_Std',
    'Original_O1_Mean', 'Original_O1_Std',
    'Original_O2_Mean', 'Original_O2_Std',
    'Original_O3_Mean', 'Original_O3_Std',
    'Original_O4_Mean', 'Original_O4_Std',
    'Original_O5_Mean', 'Original_O5_Std',
    'Original_O6_Mean', 'Original_O6_Std',
    'Original_O7_Mean', 'Original_O7_Std',
    'Original_O8_Mean', 'Original_O8_Std',
    'Ring_Mean', 'Ring_Std',
    'Ring_Q1_Mean', 'Ring_Q1_Std',
    'Ring_Q2_Mean', 'Ring_Q2_Std',
    'Ring_Q3_Mean', 'Ring_Q3_Std',
    'Ring_Q4_Mean', 'Ring_Q4_Std',
    'Ring_O1_Mean', 'Ring_O1_Std',
    'Ring_O2_Mean', 'Ring_O2_Std',
    'Ring_O3_Mean', 'Ring_O3_Std',
    'Ring_O4_Mean', 'Ring_O4_Std',
    'Ring_O5_Mean', 'Ring_O5_Std',
    'Ring_O6_Mean', 'Ring_O6_Std',
    'Ring_O7_Mean', 'Ring_O7_Std',
    'Ring_O8_Mean', 'Ring_O8_Std'
]


def create_ring_masks(contours, shape, width=DEFAULT_RING_WIDTH):
    """
    Create ring masks for all contours.
    
    Args:
        contours: List of contours
        shape: Shape of the mask (height, width)
        width: Width of the ring in pixels (default: 20)
        
    Returns:
        List of ring masks for each contour
    """
    ring_masks = []
    for contour in contours:
        # Create base mask
        base_mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(base_mask, [contour], -1, 255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width*2+1, width*2+1))
        
        # Create outer and inner ring
        outer_mask = cv2.dilate(base_mask, kernel)
        inner_mask = cv2.erode(base_mask, kernel)
        
        # Calculate ring region
        outer_ring = cv2.subtract(outer_mask, base_mask)
        inner_ring = cv2.subtract(base_mask, inner_mask)
        ring_mask = cv2.add(outer_ring, inner_ring)
        
        ring_masks.append(ring_mask)
    
    return ring_masks


def calculate_centroid(mask):
    """
    Calculate the centroid of a mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Tuple (x, y) of centroid coordinates or None if mask is empty
    """
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    return None


def split_mask_into_quadrants(mask, centroid):
    """
    Split a mask into four quadrants based on the centroid.
    
    Args:
        mask: Binary mask
        centroid: Tuple (x, y) of centroid coordinates
        
    Returns:
        List of four masks representing each quadrant or None if centroid is None
    """
    if centroid is None:
        return None
    
    cx, cy = centroid
    height, width = mask.shape
    quadrants = []
    
    # Create four quadrant masks (top right, top left, bottom left, bottom right)
    q1 = np.zeros_like(mask)
    q1[0:cy, cx:width] = mask[0:cy, cx:width]
    quadrants.append(q1)
    
    q2 = np.zeros_like(mask)
    q2[0:cy, 0:cx] = mask[0:cy, 0:cx]
    quadrants.append(q2)
    
    q3 = np.zeros_like(mask)
    q3[cy:height, 0:cx] = mask[cy:height, 0:cx]
    quadrants.append(q3)
    
    q4 = np.zeros_like(mask)
    q4[cy:height, cx:width] = mask[cy:height, cx:width]
    quadrants.append(q4)
    
    return quadrants


def split_mask_into_octants(mask, centroid):
    """
    Split a mask into eight octants (pie-shaped sectors) based on the centroid.
    
    Args:
        mask: Binary mask
        centroid: Tuple (x, y) of centroid coordinates
        
    Returns:
        List of eight masks representing each octant or None if centroid is None
    """
    if centroid is None:
        return None
    
    cx, cy = centroid
    height, width = mask.shape
    octants = []
    
    # Create angle mask
    y, x = np.ogrid[:height, :width]
    angles = np.arctan2(-(y - cy), x - cx)  # Negative sign to make y-axis positive downward
    angles = np.degrees(angles)
    angles[angles < 0] += 360  # Convert angles to 0-360 range
    
    # Define eight octant angle ranges
    angle_ranges = [(0, 45), (45, 90), (90, 135), (135, 180),
                   (180, 225), (225, 270), (270, 315), (315, 360)]
    
    # Create eight octant masks
    for start_angle, end_angle in angle_ranges:
        octant = np.zeros_like(mask)
        sector_mask = (angles >= start_angle) & (angles < end_angle)
        octant[sector_mask] = mask[sector_mask]
        octants.append(octant)
    
    return octants


def calculate_regional_statistics(img, mask):
    """
    Calculate the mean and standard deviation of pixel values in a masked region.
    
    Args:
        img: Input image
        mask: Binary mask
        
    Returns:
        Tuple (mean, std) of statistics or (0, 0) if region is empty
    """
    pixels = img[mask > 0]
    if len(pixels) > 0:
        return np.mean(pixels), np.std(pixels)
    return 0, 0


def calculate_regional_intensities(img, mask, centroid):
    """
    Calculate intensity statistics for the whole region and its quadrants and octants.
    
    Args:
        img: Input image
        mask: Binary mask
        centroid: Tuple (x, y) of centroid coordinates
        
    Returns:
        Tuple of (total_stats, quadrant_stats, octant_stats)
    """
    # Calculate total stats
    region_total_mean, region_total_std = calculate_regional_statistics(img, mask)
    
    # Calculate quadrant stats
    quad_regions = split_mask_into_quadrants(mask, centroid)
    quad_stats = []
    
    if quad_regions is not None:
        quad_stats = [calculate_regional_statistics(img, q) for q in quad_regions]
    else:
        quad_stats = [(0, 0)] * 4
    
    # Calculate octant stats
    oct_regions = split_mask_into_octants(mask, centroid)
    oct_stats = []
    
    if oct_regions is not None:
        oct_stats = [calculate_regional_statistics(img, o) for o in oct_regions]
    else:
        oct_stats = [(0, 0)] * 8
    
    return (region_total_mean, region_total_std), quad_stats, oct_stats


def create_visualization(mask_shape, contour, ring_mask, centroid, quadrants):
    """
    Create a visualization image showing the contour, ring region, and quadrants.
    
    Args:
        mask_shape: Shape of the mask (height, width)
        contour: Contour to visualize
        ring_mask: Ring mask for the contour
        centroid: Tuple (x, y) of centroid coordinates
        quadrants: List of quadrant masks
        
    Returns:
        Visualization image
    """
    # Create black background
    vis_img = np.zeros((*mask_shape, 3), dtype=np.uint8)
    
    # Draw original contour (white)
    cv2.drawContours(vis_img, [contour], -1, (255, 255, 255), -1)
    
    # Draw ring region (red)
    ring_overlay = vis_img.copy()
    ring_overlay[ring_mask > 0] = (0, 0, 255)
    
    # Add quadrant lines
    if centroid:
        cx, cy = centroid
        # Draw horizontal line (green)
        cv2.line(vis_img, (0, cy), (mask_shape[1], cy), (0, 255, 0), 2)
        # Draw vertical line (green)
        cv2.line(vis_img, (cx, 0), (cx, mask_shape[0]), (0, 255, 0), 2)
    
    # Blend original image and ring overlay
    vis_img = cv2.addWeighted(vis_img, 0.7, ring_overlay, 0.3, 0)
    
    # Label quadrants (blue text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if centroid:
        cx, cy = centroid
        cv2.putText(vis_img, 'Q1', (cx+10, cy-10), font, 0.5, (255, 0, 0), 1)
        cv2.putText(vis_img, 'Q2', (cx-30, cy-10), font, 0.5, (255, 0, 0), 1)
        cv2.putText(vis_img, 'Q3', (cx-30, cy+20), font, 0.5, (255, 0, 0), 1)
        cv2.putText(vis_img, 'Q4', (cx+10, cy+20), font, 0.5, (255, 0, 0), 1)
    
    return vis_img


def process_image(img, original_mask, ring_mask, centroid):
    """
    Process a single image to calculate all statistics.
    
    Args:
        img: Input image
        original_mask: Original binary mask
        ring_mask: Ring mask
        centroid: Tuple (x, y) of centroid coordinates
        
    Returns:
        Dictionary of statistics
    """
    # Calculate original mask statistics
    orig_total, orig_quad_stats, orig_oct_stats = calculate_regional_intensities(img, original_mask, centroid)
    
    # Calculate ring region statistics
    ring_total, ring_quad_stats, ring_oct_stats = calculate_regional_intensities(img, ring_mask, centroid)
    
    return {
        'Original_Mean': orig_total[0], 'Original_Std': orig_total[1],
        **{f'Original_Q{i+1}_Mean': quad_stat[0] for i, quad_stat in enumerate(orig_quad_stats)},
        **{f'Original_Q{i+1}_Std': quad_stat[1] for i, quad_stat in enumerate(orig_quad_stats)},
        **{f'Original_O{i+1}_Mean': oct_stat[0] for i, oct_stat in enumerate(orig_oct_stats)},
        **{f'Original_O{i+1}_Std': oct_stat[1] for i, oct_stat in enumerate(orig_oct_stats)},
        'Ring_Mean': ring_total[0], 'Ring_Std': ring_total[1],
        **{f'Ring_Q{i+1}_Mean': quad_stat[0] for i, quad_stat in enumerate(ring_quad_stats)},
        **{f'Ring_Q{i+1}_Std': quad_stat[1] for i, quad_stat in enumerate(ring_quad_stats)},
        **{f'Ring_O{i+1}_Mean': oct_stat[0] for i, oct_stat in enumerate(ring_oct_stats)},
        **{f'Ring_O{i+1}_Std': oct_stat[1] for i, oct_stat in enumerate(ring_oct_stats)}
    }


def analyze_mask(mask_path, image_sequences, base_output_folder=None, ring_width=DEFAULT_RING_WIDTH):
    """
    Analyze a mask for intensity statistics across image sequences.
    
    Args:
        mask_path: Path to the mask image
        image_sequences: Dictionary of {sequence_name: sequence_folder}
        base_output_folder: Base output folder (default: next to mask_path)
        ring_width: Width of the ring in pixels (default: 20)
        
    Returns:
        Path to the output folder
    """
    # Prepare output folder
    if base_output_folder is None:
        base_output_folder = os.path.join(os.path.dirname(mask_path), 'intensity_analysis')
    
    os.makedirs(base_output_folder, exist_ok=True)
    
    # Load and preprocess mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Cannot read mask file")
    
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get first image path to determine dimensions
    first_sequence_folder = next(iter(image_sequences.values()))
    first_image_files = sorted(os.listdir(first_sequence_folder))
    if not first_image_files:
        raise ValueError(f"No images found in {first_sequence_folder}")
    
    first_image_path = os.path.join(first_sequence_folder, first_image_files[0])
    first_img = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise ValueError(f"Cannot read image file: {first_image_path}")
    
    # Adjust mask size if needed
    img_height, img_width = first_img.shape[:2]
    mask = mask[:img_height, :img_width]
    
    # Find contours and create masks
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ring_masks = create_ring_masks(contours, mask.shape, width=ring_width)
    
    # Calculate centroids and create original masks
    centroids = []
    original_masks = []
    for contour in contours:
        original_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(original_mask, [contour], -1, 255, -1)
        original_masks.append(original_mask)
        centroids.append(calculate_centroid(original_mask))
    
    # Find the largest contour
    if contours:
        max_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    else:
        raise ValueError("No contours found in mask")
    
    # Create visualization directory
    vis_base_folder = os.path.join(base_output_folder, 'visualization')
    os.makedirs(vis_base_folder, exist_ok=True)
    
    # Create visualization for the largest contour
    vis_img = create_visualization(
        mask.shape,
        contours[max_contour_idx],
        ring_masks[max_contour_idx],
        centroids[max_contour_idx],
        split_mask_into_quadrants(original_masks[max_contour_idx], centroids[max_contour_idx])
    )
    
    # Save visualization
    vis_filename = 'mask_visualization.png'
    cv2.imwrite(os.path.join(vis_base_folder, vis_filename), vis_img)
    
    # 创建细胞区域编号标注图像
    numbered_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # 绘制原始轮廓
    for i, contour in enumerate(contours):
        # 使用不同颜色绘制每个轮廓
        color = (
            (i * 50) % 255,  # B
            (i * 87) % 255,  # G
            (i * 123) % 255  # R
        )
        cv2.drawContours(numbered_img, [contour], -1, color, -1)
        
        # 添加区域编号文本
        if centroids[i]:
            cx, cy = centroids[i]
            cv2.putText(
                numbered_img, 
                f"{i+1}", 
                (cx, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
    
    # 保存带编号的区域图
    numbered_filename = 'numbered_regions.png'
    cv2.imwrite(os.path.join(vis_base_folder, numbered_filename), numbered_img)
    
    # Process sequences in parallel
    num_processes = min(cpu_count(), len(image_sequences))
    print(f"Using {num_processes} processes for parallel processing")
    
    process_sequence_partial = partial(
        _process_sequence,
        base_output_folder=base_output_folder,
        mask_shape=mask.shape,
        contours=contours,
        ring_masks=ring_masks,
        original_masks=original_masks,
        centroids=centroids,
        max_contour_idx=max_contour_idx
    )
    
    with Pool(num_processes) as pool:
        pool.map(process_sequence_partial, image_sequences.items())
    
    print("All sequences processed")
    return base_output_folder


def _process_sequence(sequence_info, base_output_folder, mask_shape, contours, ring_masks,
                     original_masks, centroids, max_contour_idx):
    """
    Process a single image sequence.
    
    Args:
        sequence_info: Tuple of (sequence_name, sequence_folder)
        base_output_folder: Base output folder
        mask_shape: Shape of the mask
        contours: List of contours
        ring_masks: List of ring masks
        original_masks: List of original masks
        centroids: List of centroids
        max_contour_idx: Index of the largest contour
        
    Returns:
        None
    """
    sequence_name, sequence_folder = sequence_info
    sequence_folder_name = os.path.basename(sequence_folder)
    
    # Create sequence output folder
    sequence_output_folder = os.path.join(base_output_folder, sequence_folder_name)
    os.makedirs(sequence_output_folder, exist_ok=True)
    
    print(f"Processing sequence: {sequence_folder_name}")
    
    # Create CSV files for each region
    csv_files = {}
    for i in range(len(contours)):
        csv_path = os.path.join(sequence_output_folder, f'region_{i+1}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_CSV_HEADERS)
            writer.writeheader()
            csv_files[i] = csv_path
    
    # Process each image in the sequence
    image_files = sorted(os.listdir(sequence_folder))
    for frame_idx, image_file in enumerate(image_files):
        print(f"Sequence {sequence_folder_name}: Processing image {frame_idx + 1}/{len(image_files)}")
        
        image_path = os.path.join(sequence_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
            
        img = img.astype(np.float32)
        
        # Process each region
        for i, (original_mask, ring_mask, centroid) in enumerate(zip(original_masks, ring_masks, centroids)):
            if centroid is None:
                continue
            
            stats = process_image(img, original_mask, ring_mask, centroid)
            
            with open(csv_files[i], 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=DEFAULT_CSV_HEADERS)
                writer.writerow({'Frame': frame_idx + 1, **stats})


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract intensity statistics from cell regions")
    parser.add_argument("--mask_path", type=str, required=True,
                        help="Path to the processed mask image")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base directory to save results")
    parser.add_argument("--sequence_dirs", type=str, nargs='+', required=True,
                        help="List of image sequence directories")
    parser.add_argument("--sequence_names", type=str, nargs='+',
                        help="Names for the image sequences (same order as sequence_dirs)")
    parser.add_argument("--ring_width", type=int, default=DEFAULT_RING_WIDTH,
                        help=f"Width of the ring in pixels (default: {DEFAULT_RING_WIDTH})")
    
    args = parser.parse_args()
    
    # Prepare image sequences dictionary
    if args.sequence_names and len(args.sequence_names) == len(args.sequence_dirs):
        sequences = dict(zip(args.sequence_names, args.sequence_dirs))
    else:
        # Use directory names as sequence names
        sequences = {os.path.basename(d): d for d in args.sequence_dirs}
    
    # Analyze mask
    analyze_mask(args.mask_path, sequences, args.output_dir, args.ring_width)
#!/usr/bin/env python
"""
Command-line script for post-processing masks and analyzing cell intensity.

This script provides a command-line interface to the postprocess.py and analysis.py modules,
allowing users to process segmentation masks and analyze cell intensity in image sequences.
"""

import argparse
import os
import sys
from datetime import datetime

# Import local modules
from src.postprocess import process_mask, create_overlay
from src.analysis import analyze_mask


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process segmentation masks and analyze cell intensity",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process mask command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a segmentation mask (clean up, fill holes, etc.)"
    )
    process_parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the segmentation mask file"
    )
    process_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed mask (default: creates timestamped directory)"
    )
    process_parser.add_argument(
        "--original_image",
        type=str,
        default=None,
        help="Path to the original image (for overlay creation)"
    )
    process_parser.add_argument(
        "--min_area",
        type=int,
        default=3500,
        help="Minimum area for regions to keep in pixels (default: 3500)"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze cell intensity in image sequences using a processed mask"
    )
    analyze_parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the processed mask file"
    )
    analyze_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: creates directory next to mask)"
    )
    analyze_parser.add_argument(
        "--sequence_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to image sequence directories"
    )
    analyze_parser.add_argument(
        "--sequence_names",
        type=str,
        nargs="+",
        help="Names for the image sequences (must match number of directories)"
    )
    analyze_parser.add_argument(
        "--ring_width",
        type=int,
        default=20,
        help="Width of ring mask in pixels (default: 20)"
    )
    
    # Process and analyze command (combined)
    process_analyze_parser = subparsers.add_parser(
        "process_analyze",
        help="Process a mask and then analyze cell intensity"
    )
    process_analyze_parser.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to the segmentation mask file"
    )
    process_analyze_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save all results (default: creates timestamped directory)"
    )
    process_analyze_parser.add_argument(
        "--original_image",
        type=str,
        default=None,
        help="Path to the original image (for overlay creation)"
    )
    process_analyze_parser.add_argument(
        "--min_area",
        type=int,
        default=3500,
        help="Minimum area for regions to keep in pixels (default: 3500)"
    )
    process_analyze_parser.add_argument(
        "--sequence_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to image sequence directories"
    )
    process_analyze_parser.add_argument(
        "--sequence_names",
        type=str,
        nargs="+",
        help="Names for the image sequences (must match number of directories)"
    )
    process_analyze_parser.add_argument(
        "--ring_width",
        type=int,
        default=20,
        help="Width of ring mask in pixels (default: 20)"
    )
    
    if len(sys.argv) == 1:
        # No arguments provided, print help
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "process":
        # Process the mask
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join("outputs", "processed_masks", timestamp)
        
        os.makedirs(args.output_dir, exist_ok=True)
        processed_mask_path = os.path.join(args.output_dir, "processed_mask_final.png")
        
        # Process mask
        processed_mask = process_mask(
            args.mask_path,
            processed_mask_path,
            args.min_area
        )
        
        # Create overlay if original image is provided
        if args.original_image:
            overlay_path = os.path.join(args.output_dir, "mask_overlay.png")
            create_overlay(processed_mask, args.original_image, overlay_path)
            
        print(f"Mask processing completed. Results saved to: {args.output_dir}")
        
    elif args.command == "analyze":
        # Analyze cell intensity
        if args.output_dir is None:
            args.output_dir = os.path.join(os.path.dirname(args.mask_path), "intensity_analysis")
        
        # Prepare image sequences dictionary
        if args.sequence_names and len(args.sequence_names) == len(args.sequence_dirs):
            sequences = dict(zip(args.sequence_names, args.sequence_dirs))
        else:
            # Use directory names as sequence names
            sequences = {os.path.basename(d): d for d in args.sequence_dirs}
        
        # Analyze mask
        output_dir = analyze_mask(
            args.mask_path,
            sequences,
            args.output_dir,
            args.ring_width
        )
        
        print(f"Analysis completed. Results saved to: {output_dir}")
        
    elif args.command == "process_analyze":
        # Process and analyze
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = os.path.join("outputs", "cell_analysis", timestamp)
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        mask_dir = os.path.join(args.output_dir, "processed_mask")
        os.makedirs(mask_dir, exist_ok=True)
        analysis_dir = os.path.join(args.output_dir, "intensity_analysis")
        
        # Process mask
        processed_mask_path = os.path.join(mask_dir, "processed_mask_final.png")
        processed_mask = process_mask(
            args.mask_path,
            processed_mask_path,
            args.min_area
        )
        
        # Create overlay if original image is provided
        if args.original_image:
            overlay_path = os.path.join(mask_dir, "mask_overlay.png")
            create_overlay(processed_mask, args.original_image, overlay_path)
        
        # Prepare image sequences dictionary
        if args.sequence_names and len(args.sequence_names) == len(args.sequence_dirs):
            sequences = dict(zip(args.sequence_names, args.sequence_dirs))
        else:
            # Use directory names as sequence names
            sequences = {os.path.basename(d): d for d in args.sequence_dirs}
        
        # Analyze mask
        analyze_mask(
            processed_mask_path,
            sequences,
            analysis_dir,
            args.ring_width
        )
        
        print(f"Processing and analysis completed. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

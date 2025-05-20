"""
Entry point to run the UNetSLSM module from any location.
"""

import os
import sys
import argparse

# Add the current directory to the path so we can import the src package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main as run_main


def parse_args():
    parser = argparse.ArgumentParser(description="UNetSLSM Cell Segmentation Tool")
    parser.add_argument("--config", type=str, default="config/test_config.yaml", 
                        help="Path to config file")
    parser.add_argument("--test-only", action="store_true", 
                        help="Only test the module structure without running inference")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force the use of CPU even if GPU is available")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.test_only:
        # Import and run the structure test
        print("Running structure tests only...")
        try:
            from test_structure import test_directory_structure, test_model_without_loading
            from test_structure import test_forward_pass, test_dataset, test_import_modules
            from test_structure import test_config_loading
            
            # Run tests
            test_directory_structure()
            model = test_model_without_loading()
            test_forward_pass(model)
            test_dataset()
            test_import_modules()
            test_config_loading()
            
            print("\nStructure tests completed. Run without --test-only to perform inference.")
        except ImportError as e:
            print(f"Error importing test modules: {e}")
    else:        # Run the main function with the specified config
        run_main(args.config, args.force_cpu)

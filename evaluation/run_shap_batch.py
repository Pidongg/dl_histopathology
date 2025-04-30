# runs SHAP analysis on a batch of images with a YOLO model.
import os
import argparse
from shap_analysis import analyze_all_detections

def main():
    # Dictionary mapping filenames to list of superpixel sizes
    filenames_to_superpixel_size = {
        "703488 [d=0.98892,x=101582,y=53165,w=507,h=506]": [32],
        "703488 [d=0.98892,x=114873,y=37975,w=507,h=506]": [16],
        "703488 [d=0.98892,x=120095,y=51266,w=506,h=506]": [16, 8],
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=25633,y=40348,w=506,h=506]": [16],
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=29905,y=36551,w=506,h=506]": [16],
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=30380,y=39399,w=506,h=506]": [8],
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=34652,y=29905,w=506,h=506]": [32],
        "771913 [d=0.98892,x=56013,y=37500,w=506,h=506]": [16],
        "771913 [d=0.98892,x=83070,y=39399,w=506,h=506]": [8],
        "771913 [d=0.98892,x=83544,y=39873,w=507,h=507]": [16],
    }
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SHAP analysis on multiple files with specified superpixel sizes")
    parser.add_argument("-pt", "--model_path", required=True, help="Path to model .pt file")
    parser.add_argument("-base", "--base_path", default="M:/Unused/TauCellDL", help="Base path for images and labels")
    parser.add_argument("-samples", "--num_samples", type=int, default=1000, help="Number of samples for SHAP analysis")
    parser.add_argument("-conf", "--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("-out", "--output_dir", default="./shap_results", help="Output directory for visualizations")
    parser.add_argument("-class_names", "--class_names", type=str, default="tau_positive", help="Comma-separated list of class names")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file with its specified superpixel sizes
    for filename, superpixel_sizes in filenames_to_superpixel_size.items():
        # Extract the sample ID (e.g., "703488") from the filename
        sample_id = filename.split()[0].split('.')[0]
        
        # Construct the image path
        image_path = os.path.join(args.base_path, "images", "test", sample_id, f"{filename}.png")
        
        # Construct the ground truth path
        gt_label_dir = os.path.join(args.base_path, "labels", "test", sample_id)
        
        # Process each superpixel size for this file
        for spx_size in superpixel_sizes:
            print(f"\nProcessing {filename} with superpixel size {spx_size}")
            
            # Create output subdirectory for this file and superpixel size
            file_output_dir = os.path.join(args.output_dir, f"{sample_id}_{spx_size}px")
            os.makedirs(file_output_dir, exist_ok=True)
            
            try:
                # Run SHAP analysis
                analyze_all_detections(
                    model_path=args.model_path,
                    image_path=image_path,
                    super_pixel_size=spx_size,
                    num_samples=args.num_samples,
                    conf_thresh=args.conf_thresh,
                    output_dir=file_output_dir,
                    gt_label_dir=gt_label_dir,
                    class_names=args.class_names.split(',')
                )
                print(f"Successfully processed {filename} with superpixel size {spx_size}")
            except Exception as e:
                print(f"Error processing {filename} with superpixel size {spx_size}: {str(e)}")

if __name__ == "__main__":
    main() 
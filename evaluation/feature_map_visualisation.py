# Visualizes feature maps of a YOLO model on a batch of images (same batch as in shap analysis).
import os
import argparse
from ultralytics import YOLO
from PIL import Image

def main():
    # List of filenames to process (same as in run_shap_batch.py)
    filenames_to_process = [
        "703488 [d=0.98892,x=101582,y=53165,w=507,h=506]",
        "703488 [d=0.98892,x=114873,y=37975,w=507,h=506]",
        "703488 [d=0.98892,x=120095,y=51266,w=506,h=506]",
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=25633,y=40348,w=506,h=506]",
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=29905,y=36551,w=506,h=506]",
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=30380,y=39399,w=506,h=506]",
        "747331.svs_training_test_tanrada_5%_2 [d=0.98892,x=34652,y=29905,w=506,h=506]",
        "771913 [d=0.98892,x=56013,y=37500,w=506,h=506]",
        "771913 [d=0.98892,x=83070,y=39399,w=506,h=506]",
        "771913 [d=0.98892,x=83544,y=39873,w=507,h=507]",
    ]
    
    parser = argparse.ArgumentParser(description="Run feature map visualization on multiple files")
    parser.add_argument("-pt", "--model_path", required=True, help="Path to model .pt file")
    parser.add_argument("-base", "--base_path", default="M:/Unused/TauCellDL", help="Base path for images")
    parser.add_argument("-out", "--output_dir", default="./feature_map_results", help="Output directory for visualizations")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(args.model_path)
    
    # Process each file
    for filename in filenames_to_process:
        # Extract the sample ID (e.g., "703488") from the filename
        sample_id = filename.split('.')[0].split()[0]
        
        # Construct the image path
        image_path = os.path.join(args.base_path, "images", "test", sample_id, f"{filename}.png")
        
        # Create output subdirectory for this file
        file_output_dir = os.path.join(args.output_dir, sample_id)
        os.makedirs(file_output_dir, exist_ok=True)
        
        try:
            # Run inference with visualize=True to generate feature map visualizations
            results = model(image_path, visualize=True)
            
            # Also save the detection output
            img_array = results[0].plot()
            img_with_dets = Image.fromarray(img_array[..., ::-1])
            img_with_dets.save(os.path.join(file_output_dir, f"{os.path.basename(image_path).split('.')[0]}_detections.png"))
            
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main() 
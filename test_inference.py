import glob
import os
import torch
import numpy as np
from depth_anything_3.api import DepthAnything3

def test_inference():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Use a smaller model for faster test
    model_name = "depth-anything/DA3-SMALL"
    print(f"Loading model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    
    # Create a dummy image for testing if no assets are found
    example_path = "assets/examples/SOH"
    if not os.path.exists(example_path):
        os.makedirs(example_path, exist_ok=True)
        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        dummy_img.save(os.path.join(example_path, "dummy.png"))
    
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    print(f"Found {len(images)} images.")
    
    if len(images) == 0:
        print("No images found to test.")
        return

    print("Running inference...")
    prediction = model.inference(images)
    
    print("Inference results:")
    print(f"Processed images shape: {prediction.processed_images.shape}")
    print(f"Depth shape: {prediction.depth.shape}")
    print(f"Confidence shape: {prediction.conf.shape}")

if __name__ == "__main__":
    test_inference()

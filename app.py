import sys
import json
import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_and_preprocess(image_path, target_size):
    """Load image, convert to grayscale, and resize"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')  # Remove alpha channel if present
        img = img.convert('L')    # Convert to grayscale
        img = img.resize(target_size)
        return np.array(img)
    except Exception as e:
        print(f"Skipping {image_path}: {str(e)}")
        return None

def calculate_similarity(new_image_path, folder_path):
    # Load reference image
    ref_img = Image.open(new_image_path)
    target_size = ref_img.size  # Use original size of new_image
    ref_array = load_and_preprocess(new_image_path, target_size)
    
    results = {}
    
    # Compare with each image in folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            cmp_array = load_and_preprocess(file_path, target_size)
            if cmp_array is not None:
                try:
                    score = ssim(ref_array, cmp_array, 
                               data_range=255)  # 8-bit images (0-255)
                    results[filename] = round(score, 2)
                except Exception as e:
                    print(f"Error comparing {filename}: {str(e)}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app.py <new_image> <image_folder>")
        sys.exit(1)
    
    new_image_path = sys.argv[1]
    folder_path = sys.argv[2]
    
    similarity_scores = calculate_similarity(new_image_path, folder_path)
    
    # Save results to JSON
    with open('similarity_results.json', 'w') as f:
        json.dump(similarity_scores, f, indent=2)
    
    print("Comparison complete. Results saved to similarity_results.json")
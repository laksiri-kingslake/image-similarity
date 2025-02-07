import sys
import json
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Initialize model once
MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess(img_path):
    """Load and preprocess image for ResNet50"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # ResNet input size
        x = image.img_to_array(img)
        x = preprocess_input(x)
        return x
    except Exception as e:
        print(f"Skipping {img_path}: {str(e)}")
        return None

def img_to_embedding(img_array):
    """Convert preprocessed image to embedding vector"""
    return MODEL.predict(np.expand_dims(img_array, axis=0))[0]

def calculate_similarities(new_img_path, folder_path):
    # Process reference image
    ref_array = load_and_preprocess(new_img_path)
    if ref_array is None:
        raise ValueError("Failed to process reference image")
    ref_embedding = img_to_embedding(ref_array)
    
    results = {}
    
    # Process comparison images
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            cmp_array = load_and_preprocess(file_path)
            if cmp_array is not None:
                try:
                    cmp_embedding = img_to_embedding(cmp_array)
                    similarity = cosine_similarity(
                        [ref_embedding], 
                        [cmp_embedding]
                    )[0][0]
                    results[filename] = round(float(similarity), 4)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python app-ml.py <new_image> <image_folder>")
        sys.exit(1)
    
    new_image_path = sys.argv[1]
    folder_path = sys.argv[2]
    
    try:
        similarities = calculate_similarities(new_image_path, folder_path)
        with open('vector_similarities.json', 'w') as f:
            json.dump(similarities, f, indent=2)
        print("Results saved to vector_similarities.json")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
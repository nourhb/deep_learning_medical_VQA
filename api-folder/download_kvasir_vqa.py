import os
import pandas as pd
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset
import requests
import json

# Directory setup
DATA_DIR = "kvasir_data"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

print("Downloading Kvasir-VQA dataset...")

try:
    # Load dataset with specific configuration
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(
        "SimulaMet-HOST/Kvasir-VQA",
        split="train",
        trust_remote_code=True
    )
    
    # Convert to DataFrame
    print("Processing dataset...")
    df = pd.DataFrame(dataset)
    print(f"Found {len(df)} rows in the dataset")
    
    # Save as CSV
    print(f"Saving kvasir_vqa_full.csv...")
    df.to_csv(os.path.join(DATA_DIR, "kvasir_vqa_full.csv"), index=False)
    
    # Download images
    print("Downloading images...")
    for i, row in df.iterrows():
        try:
            # Get image URL from the source field
            img_url = row['source']
            img_id = row['img_id']
            
            # Download and save image
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            img = Image.open(BytesIO(img_response.content))
            img.save(os.path.join(IMG_DIR, f"{img_id}.jpg"))
            
            if (i+1) % 50 == 0:
                print(f"Downloaded {i+1} images...")
        except Exception as e:
            print(f"Error downloading image {i}: {str(e)}")
            continue
    
    print("Download complete!")

except Exception as e:
    print(f"Error: {str(e)}")
    print("Please make sure you have the required packages installed:")
    print("pip install datasets pandas pillow requests") 
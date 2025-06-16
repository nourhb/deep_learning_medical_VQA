import torch
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
from transformers import AutoTokenizer
import torchvision.transforms as T
import gc

def create_dataset_chunks():
    print("Setting up dataset processing...")
    
    # Create directories
    os.makedirs("dataset_chunks", exist_ok=True)
    os.makedirs("dataset_chunks/images", exist_ok=True)
    os.makedirs("dataset_chunks/processed", exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # Image transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset in streaming mode
    print("Loading Kvasir-VQA dataset...")
    dataset = load_dataset("SimulaMet-HOST/Kvasir-VQA", streaming=True)
    
    # Process training data
    print("Processing training data...")
    train_data = []
    train_answers = set()
    
    for idx, item in enumerate(tqdm(dataset['train'])):
        try:
            # Process image
            image = Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
            image_tensor = transform(image)
            
            # Save image tensor
            image_path = f"dataset_chunks/images/train_{idx}.pt"
            torch.save(image_tensor, image_path)
            
            # Process question
            tokens = tokenizer(
                item['question'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Store metadata
            train_data.append({
                'image_path': image_path,
                'input_ids': tokens['input_ids'].squeeze(0).numpy(),
                'attention_mask': tokens['attention_mask'].squeeze(0).numpy(),
                'answer': item['answer']
            })
            
            train_answers.add(item['answer'])
            
            # Save in chunks of 1000
            if (idx + 1) % 1000 == 0:
                chunk_idx = (idx + 1) // 1000
                np.save(f"dataset_chunks/processed/train_chunk_{chunk_idx}.npy", train_data)
                train_data = []
                gc.collect()
                
        except Exception as e:
            print(f"Error processing training sample {idx}: {e}")
            continue
    
    # Save remaining training data
    if train_data:
        chunk_idx = (idx + 1) // 1000 + 1
        np.save(f"dataset_chunks/processed/train_chunk_{chunk_idx}.npy", train_data)
    
    # Process validation data
    print("Processing validation data...")
    val_data = []
    
    for idx, item in enumerate(tqdm(dataset['validation'])):
        try:
            # Process image
            image = Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
            image_tensor = transform(image)
            
            # Save image tensor
            image_path = f"dataset_chunks/images/val_{idx}.pt"
            torch.save(image_tensor, image_path)
            
            # Process question
            tokens = tokenizer(
                item['question'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Store metadata
            val_data.append({
                'image_path': image_path,
                'input_ids': tokens['input_ids'].squeeze(0).numpy(),
                'attention_mask': tokens['attention_mask'].squeeze(0).numpy(),
                'answer': item['answer']
            })
            
            # Save in chunks of 500
            if (idx + 1) % 500 == 0:
                chunk_idx = (idx + 1) // 500
                np.save(f"dataset_chunks/processed/val_chunk_{chunk_idx}.npy", val_data)
                val_data = []
                gc.collect()
                
        except Exception as e:
            print(f"Error processing validation sample {idx}: {e}")
            continue
    
    # Save remaining validation data
    if val_data:
        chunk_idx = (idx + 1) // 500 + 1
        np.save(f"dataset_chunks/processed/val_chunk_{chunk_idx}.npy", val_data)
    
    # Create and save label map
    unique_answers = sorted(list(train_answers))
    answer_to_index = {ans: idx for idx, ans in enumerate(unique_answers)}
    
    with open("dataset_chunks/label_map.json", "w") as f:
        json.dump(answer_to_index, f, indent=2)
    
    print("Dataset processing complete!")

if __name__ == "__main__":
    create_dataset_chunks() 
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForVision2Seq
)
from datasets import load_dataset
import logging
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from tqdm import tqdm
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_disk_space(path, required_gb=5):
    """Check if there's enough disk space available."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)  # Convert to GB
    logger.info(f"Free disk space: {free_gb}GB")
    if free_gb < required_gb:
        raise RuntimeError(f"Not enough disk space. Required: {required_gb}GB, Available: {free_gb}GB")
    return True

class MemoryEfficientVQADataset(Dataset):
    def __init__(self, split="raw", max_samples=None, cache_dir=None):
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "dataset_cache")
        
        # Initialize model components
        logger.info("Initializing model components...")
        try:
            self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-base")
            logger.info("Successfully loaded processor and tokenizer")
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
            
        self.metadata = []
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check disk space before proceeding
        check_disk_space(self.cache_dir)
        
        self._create_metadata()
        
    def _create_metadata(self):
        try:
            logger.info(f"Loading dataset split: {self.split}")
            logger.info(f"Cache directory: {self.cache_dir}")
            
            # First try to load the dataset without streaming to check if it exists
            try:
                dataset_info = load_dataset("SimulaMet-HOST/Kvasir-VQA", split=self.split, cache_dir=self.cache_dir)
                logger.info(f"Dataset info: {dataset_info}")
            except Exception as e:
                logger.error(f"Error loading dataset info: {str(e)}")
                raise
            
            # Now load with streaming
            dataset = load_dataset(
                "SimulaMet-HOST/Kvasir-VQA",
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=True
            )
            
            # Convert streaming dataset to list for metadata
            logger.info("Processing dataset metadata...")
            sample_count = 0
            error_count = 0
            
            for idx, item in enumerate(tqdm(dataset)):
                if self.max_samples and idx >= self.max_samples:
                    break
                    
                try:
                    # Log the first few items for debugging
                    if idx < 3:
                        logger.info(f"Sample {idx} structure: {item}")
                    
                    # Validate image URL
                    if not item.get('image_url'):
                        logger.warning(f"Missing image URL for item {idx}")
                        error_count += 1
                        continue
                        
                    # Validate question and answer
                    if not item.get('question') or not item.get('answer'):
                        logger.warning(f"Missing question or answer for item {idx}")
                        error_count += 1
                        continue
                    
                    # Try to load the image to validate the URL
                    try:
                        response = requests.head(item['image_url'], timeout=5)
                        if response.status_code != 200:
                            logger.warning(f"Invalid image URL for item {idx}: {item['image_url']}")
                            error_count += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Error checking image URL for item {idx}: {str(e)}")
                        error_count += 1
                        continue
                    
                    self.metadata.append({
                        'image_url': item['image_url'],
                        'question': item['question'],
                        'answer': item['answer']
                    })
                    sample_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    error_count += 1
                    continue
            
            if not self.metadata:
                raise ValueError(f"No valid samples found in dataset. Processed {sample_count} samples with {error_count} errors.")
                
            logger.info(f"Successfully loaded {len(self.metadata)} samples")
            logger.info(f"Encountered {error_count} errors during processing")
            
        except Exception as e:
            logger.error(f"Error creating metadata: {str(e)}")
            raise
            
    def _load_image(self, image_url):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            logger.error(f"Error loading image from {image_url}: {str(e)}")
            return None
            
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        try:
            item = self.metadata[idx]
            image = self._load_image(item['image_url'])
            
            if image is None:
                # Return a placeholder if image loading fails
                return {
                    'pixel_values': torch.zeros((3, 224, 224)),
                    'input_ids': torch.zeros(20, dtype=torch.long),
                    'attention_mask': torch.zeros(20, dtype=torch.long),
                    'labels': torch.zeros(20, dtype=torch.long)
                }
            
            # Process image and text
            inputs = self.processor(
                images=image,
                text=item['question'],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=20
            )
            
            # Add answer as labels
            answer_inputs = self.tokenizer(
                item['answer'],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=20
            )
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': answer_inputs['input_ids'].squeeze(0)
            }
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for idx {idx}: {str(e)}")
            # Return a placeholder in case of error
            return {
                'pixel_values': torch.zeros((3, 224, 224)),
                'input_ids': torch.zeros(20, dtype=torch.long),
                'attention_mask': torch.zeros(20, dtype=torch.long),
                'labels': torch.zeros(20, dtype=torch.long)
            }

def collate_fn(batch):
    try:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        raise

def train_model():
    try:
        # Check disk space before starting
        check_disk_space(os.path.dirname(__file__))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and processor
        logger.info("Loading model...")
        try:
            model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-vqa-base")
            model = model.to(device)
            logger.info("Successfully loaded model")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Load datasets with smaller batch size and fewer samples
        logger.info("Loading training dataset...")
        train_dataset = MemoryEfficientVQADataset(
            split="raw",
            max_samples=100  # Further reduced sample size for testing
        )
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
            
        logger.info(f"Training dataset size: {len(train_dataset)}")
            
        logger.info("Loading validation dataset...")
        val_dataset = MemoryEfficientVQADataset(
            split="raw",
            max_samples=20  # Further reduced sample size for testing
        )
        
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")
            
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Create data loaders with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # Further reduced batch size
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,  # Further reduced batch size
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                try:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Clear memory
                    del outputs
                    del loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    try:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()
                        
                        # Clear memory
                        del outputs
                        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(os.path.dirname(__file__), f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 
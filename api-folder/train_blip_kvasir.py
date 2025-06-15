
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForQuestionAnswering, BlipProcessor, TrainingArguments, Trainer
import torch

class KvasirVQADataset(Dataset):
    def __init__(self, csv_path, img_folder, processor, max_length=32):
        self.data = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_folder, f"{row['img_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        question = str(row['question'])
        answer = str(row['answer'])
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        labels = self.processor.tokenizer(answer, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).input_ids
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze(0)
        return inputs

if __name__ == "__main__":
    # Paths
    csv_path = "kvasir_data/sample_100.csv"
    img_folder = "kvasir_data/images"
    output_dir = "saved_models/blip_kvasir_vqa"

    # Model and processor
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    # Dataset
    dataset = KvasirVQADataset(csv_path, img_folder, processor)
    # For demo, use a subset. Remove [:1000] for full training.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,  # Increase for real training
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    trainer.train()

    # Save model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}") 
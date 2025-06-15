import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BlipForQuestionAnswering, BlipProcessor, TrainingArguments, Trainer
import torch
import json

class KvasirVQADataset(Dataset):
    def __init__(self, csv_path, processor, max_length=32):
        self.data = pd.read_csv(csv_path)
        self.processor = processor
        self.max_length = max_length
        # Filter to only include relevant answers (no locations)
        with open("label_map_cleaned.json", "r") as f:
            self.label_map = json.load(f)
        self.data = self.data[self.data['answer'].isin(self.label_map.keys())]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        question = str(row['question'])
        answer = str(row['answer'])
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        labels = self.processor.tokenizer(answer, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).input_ids
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze(0)
        return inputs

if __name__ == "__main__":
    csv_path = "kvasir_data/sample_100.csv"  # Update to your full dataset if available
    output_dir = "saved_models/blip_kvasir_vqa_cleaned"

    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    dataset = KvasirVQADataset(csv_path, processor)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"BLIP model and processor saved to {output_dir}") 
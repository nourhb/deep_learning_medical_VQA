from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor, TrainingArguments, Trainer
import torch

# Load Kvasir-VQA dataset
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
# For testing, use a subset. Remove .select for full training.
train_data = ds['raw'].select(range(1000))
val_data = ds['raw'].select(range(1000, 1200))

# Load BLIP model and processor
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

def preprocess(example):
    inputs = processor(images=example['image'], text=example['question'], return_tensors="pt", padding=True)
    inputs['labels'] = processor.tokenizer(example['answer'], return_tensors="pt", padding=True).input_ids
    return {
        'pixel_values': inputs['pixel_values'].squeeze(),
        'input_ids': inputs['input_ids'].squeeze(),
        'attention_mask': inputs['attention_mask'].squeeze(),
        'labels': inputs['labels'].squeeze()
    }

train_data = train_data.map(preprocess)
val_data = val_data.map(preprocess)

columns = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
train_data.set_format(type='torch', columns=columns)
val_data.set_format(type='torch', columns=columns)

training_args = TrainingArguments(
    output_dir="./blip-vqa-kvasir",
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
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

# Save your model and processor
model.save_pretrained("./my_kvasir_vqa_model")
processor.save_pretrained("./my_kvasir_vqa_model")

d_path = "./kvasir_data"
df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
df.to_csv(f"{d_path}/metadata.csv", index=False)

import os
os.makedirs(f"{d_path}/images", exist_ok=True)

for i, row in df.groupby('img_id').nth(0).iterrows():
    image = ds['raw'][i]['image'].save(f"{d_path}/images/{row['img_id']}.jpg")
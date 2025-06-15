from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor, TrainingArguments, Trainer
import torch

# Load dataset
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
train_data = ds['raw'].train_test_split(test_size=0.1)['train']
val_data = ds['raw'].train_test_split(test_size=0.1)['test']

# Load model and processor
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

def preprocess(example):
    inputs = processor(images=example['image'], text=example['question'], return_tensors="pt", padding=True)
    inputs['labels'] = processor.tokenizer(example['answer'], return_tensors="pt", padding=True).input_ids
    return inputs

train_data = train_data.map(preprocess, batched=True)
val_data = val_data.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./blip-vqa-kvasir",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

model.save_pretrained("./my_kvasir_vqa_model")
processor.save_pretrained("./my_kvasir_vqa_model")

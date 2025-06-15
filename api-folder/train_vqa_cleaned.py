import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from transformers import BertTokenizer
import json
import os
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

# Load cleaned label map and create continuous indices
with open("label_map_cleaned.json", "r") as f:
    label_map_raw = json.load(f)

# Create new continuous mapping
unique_answers = sorted(set(label_map_raw.keys()))
answer_to_index = {ans: idx for idx, ans in enumerate(unique_answers)}
index_to_answer = {str(idx): ans for ans, idx in answer_to_index.items()}

# Save the new mapping for later use
with open("label_map_continuous.json", "w") as f:
    json.dump(answer_to_index, f, indent=2)

# Dummy dataset class (replace with your real data loader)
class VQADataset(Dataset):
    def __init__(self, csv_path, img_folder, tokenizer, label_map):
        import pandas as pd
        self.data = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Filter to only include answers in label_map
        self.data = self.data[self.data['answer'].isin(label_map.keys())]
        print(f"Dataset size after filtering: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_folder, row['img_id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        question = row['question']
        answer = row['answer']
        image_tensor = self.transform(image)
        tokens = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        label = self.label_map[answer]
        return image_tensor, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), label

# Model definition (same as your model.py)
class VQAModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        self.mobilenet.classifier[1] = torch.nn.Identity()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280 + 768, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_labels)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.mobilenet(image)
        text_features = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        combined = torch.cat((image_features, text_features), dim=1)
        return self.fc(combined)

if __name__ == "__main__":
    # Paths
    csv_path = "kvasir_data/kvasir_vqa_full.csv"
    img_folder = "kvasir_data/images"
    model_save_path = "saved_models/best_vqa_model_cleaned.pth"
    pretrained_path = "saved_models/best_vqa_model.pth"

    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
    dataset = VQADataset(csv_path, img_folder, tokenizer, answer_to_index)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Number of unique labels: {len(answer_to_index)}")
    model = VQAModel(num_labels=len(answer_to_index))
    model = model.to("cpu")

    # Fine-tune from checkpoint if available
    if os.path.exists(pretrained_path):
        print(f"Loading weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        # Remove the last layer weights
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.3")}
        model.load_state_dict(state_dict, strict=False)
        print("Loaded all layers except the last fully connected layer.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Split into train/val
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=4, sampler=val_sampler)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (image, input_ids, attention_mask, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(image, input_ids, attention_mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for image, input_ids, attention_mask, label in val_loader:
                output = model(image, input_ids, attention_mask)
                _, pred = torch.max(output, 1)
                correct += (pred == label).sum().item()
                total += label.size(0)
                all_labels.extend(label.tolist())
                all_preds.extend(pred.tolist())
        val_acc = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Per-class accuracy
        class_correct = defaultdict(int)
        class_total = Counter(all_labels)
        for l, p in zip(all_labels, all_preds):
            if l == p:
                class_correct[l] += 1
        print("Per-class accuracy:")
        for label, total_count in class_total.items():
            acc = class_correct[label] / total_count if total_count > 0 else 0
            ans = index_to_answer[str(label)]
            print(f"  {ans}: {acc:.2f} ({class_correct[label]}/{total_count})")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_map": answer_to_index,
        "index_to_answer": index_to_answer
    }, model_save_path)
    print(f"Model saved to {model_save_path}") 
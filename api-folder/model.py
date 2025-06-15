import torch
from PIL import Image
import torchvision.transforms as T
from transformers import BertTokenizer
import json

# Load label_map_continuous.json
with open("label_map_continuous.json", "r") as f:
    label_map_raw = json.load(f)

# Invert the mapping: index (as str) → answer
index_to_answer = {str(v): k for k, v in label_map_raw.items()}

# Model definition
class VQAModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
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

    def forward(self, image, tokens):
        image_features = self.mobilenet(image)
        text_features = self.bert(tokens['input_ids'], attention_mask=tokens['attention_mask']).pooler_output
        combined = torch.cat((image_features, text_features), dim=1)
        return self.fc(combined)

# Load model
model_path = "saved_models/best_vqa_model_cleaned.pth"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model = VQAModel(num_labels=len(label_map_raw))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(torch.device("cpu")).eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

# Image preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image, question: str):
    if image is None or not question.strip():
        return "Image ou question manquante"
    image_tensor = transform(image).unsqueeze(0).to("cpu")
    tokens = tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokens = {key: val.to("cpu") for key, val in tokens.items()}
    with torch.no_grad():
        output = model(image_tensor, tokens)
        print(f"[DEBUG] Raw model output: {output}")
        _, pred = torch.max(output, 1)
        print(f"[DEBUG] Predicted index: {pred.item()}")
        mapped_answer = index_to_answer.get(str(pred.item()), "❓ Inconnu")
        print(f"[DEBUG] Mapped answer: {mapped_answer}")
    return mapped_answer
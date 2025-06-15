import torch
from train_vqa_cleaned import VQADataset, VQAModel, answer_to_index, index_to_answer
from transformers import BertTokenizer
from collections import Counter, defaultdict

# Paths
csv_path = "kvasir_data/kvasir_vqa_full.csv"  # or your test CSV
img_folder = "kvasir_data/images"
model_path = "saved_models/best_vqa_model_cleaned.pth"

# Load tokenizer, dataset, and model
print("Loading test dataset...")
tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
test_dataset = VQADataset(csv_path, img_folder, tokenizer, answer_to_index)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

print("Loading model...")
model = VQAModel(num_labels=len(answer_to_index))
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Test loop
correct = 0
total = 0
all_labels = []
all_preds = []
for image, input_ids, attention_mask, label in test_loader:
    with torch.no_grad():
        output = model(image, input_ids, attention_mask)
        _, pred = torch.max(output, 1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        all_labels.extend(label.tolist())
        all_preds.extend(pred.tolist())

print(f"Test Accuracy: {correct / total:.4f}")

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
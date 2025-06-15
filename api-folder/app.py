from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json
import torch
from model import VQAModel, predict
from model_blip import predict_blip

app = Flask(__name__)
CORS(app)

# Load the cleaned VQA model and label map
with open("label_map_continuous.json", "r") as f:
    label_map = json.load(f)
index_to_answer = {str(v): k for k, v in label_map.items()}

model = VQAModel(num_labels=len(label_map))
checkpoint = torch.load("saved_models/best_vqa_model_cleaned.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# List of location/direction answers to filter out
location_labels = set([
    "center", "centerleft", "centerright", "lowercenter", "lowerleft", "lowerright",
    "uppercenter", "upperleft", "upperright"
])

def is_location_answer(answer):
    # If answer is a single location or a space-separated list of locations
    tokens = answer.lower().split()
    return all(token in location_labels for token in tokens)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400
    
    image_file = request.files['image']
    question = request.form['question']
    model_type = request.form.get('model_type', 'vqa')  # Default to VQA model
    
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        print(f"[DEBUG] Using model: {model_type}")
        print(f"[DEBUG] Label map: label_map_continuous.json")
        print(f"[DEBUG] Question: {question}")
        # Always use cleaned VQA model first
        answer = predict(image, question)
        print(f"[DEBUG] VQA answer: {answer}")
        # Fallback to BLIP if answer is a location or not relevant
        if is_location_answer(answer) or answer.strip() == '' or answer.lower() == 'unknown':
            print("[DEBUG] Fallback to BLIP model")
            answer = predict_blip(image, question)
        print(f"[DEBUG] Final answer: {answer}")
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image
import torch
import os

# Try to use local fine-tuned model, else fallback to open-source BLIP
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models', 'blip_kvasir_vqa_cleaned')
DEFAULT_MODEL = "Salesforce/blip-vqa-base"

# Only load model/processor when needed
_blip_model = None
_blip_processor = None

def predict_blip(image: Image.Image, question: str):
    global _blip_model, _blip_processor
    if _blip_model is None or _blip_processor is None:
        if os.path.isdir(LOCAL_MODEL_DIR):
            print("[DEBUG] Loading local fine-tuned BLIP model...")
            _blip_model = BlipForQuestionAnswering.from_pretrained(LOCAL_MODEL_DIR)
            _blip_processor = BlipProcessor.from_pretrained(LOCAL_MODEL_DIR)
        else:
            print("[DEBUG] Loading open-source BLIP model (Salesforce/blip-vqa-base)...")
            _blip_model = BlipForQuestionAnswering.from_pretrained(DEFAULT_MODEL)
            _blip_processor = BlipProcessor.from_pretrained(DEFAULT_MODEL)
        _blip_model.eval()
    inputs = _blip_processor(images=image, text=question, return_tensors="pt")
    with torch.no_grad():
        out = _blip_model.generate(**inputs)
    answer = _blip_processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return answer 
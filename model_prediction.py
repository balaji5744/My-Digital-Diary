from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model from your local directory
# Make sure your pytorch_model.bin is in this folder alongside the config files!
MODEL_DIR = "./my_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure your model weight file (e.g., pytorch_model.bin) is in the my_model folder.")

# Emotion to Emoji Mapping based on your config.json
EMOJI_DICT = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😮"
}

def predict_emotion(diary_text):
    if not diary_text.strip():
        return "neutral", "😐"
        
    # Tokenize the input text
    inputs = tokenizer(
        diary_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    
    # Map the class ID back to the label name using your config mapping
    label2name = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    predicted_label = label2name.get(predicted_class_id, "neutral")
    
    emoji = EMOJI_DICT.get(predicted_label, "😐")
    
    return predicted_label, emoji
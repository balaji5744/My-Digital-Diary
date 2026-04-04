import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json

# --- 1. CONFIGURATION & SETUP ---
# Load configuration mapping
with open("config.json", "r") as f:
    config = json.load(f)

# Map the text labels to emojis
EMOJI_MAP = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

# --- 2. YOUR EXACT MODEL BLUEPRINT FROM COLAB ---
class BertweetSupConClassifier(nn.Module):
    def __init__(self, model_name, num_labels, proj_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = outputs.last_hidden_state[:, 0]
        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)
        projected = self.projection(cls_embedding)
        projected = F.normalize(projected, p=2, dim=1)

        # Returns both logits (for prediction) and projected features (for contrastive loss)
        return logits, projected

# --- 3. CACHING & LOADING ---
# We use @st.cache_resource so the model only loads once when the app starts
@st.cache_resource
def load_model_and_tokenizer():
    # BERTweet requires normalization=True
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], normalization=True)
    
    # Initialize your specific model structure using the config
    model = BertweetSupConClassifier(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        proj_dim=config["proj_dim"],
        dropout=config["dropout"]
    )
    
    # Load the trained weights onto the CPU (weights_only=True for safety)
    model.load_state_dict(torch.load("bertweet_supcon_last_epoch_state_dict.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval() # Set to evaluation mode
    
    return model, tokenizer

# Load the assets
model, tokenizer = load_model_and_tokenizer()

# --- 4. STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Emotion Analyzer", page_icon="🎭")

st.title("🎭 Text Emotion Analyzer")
st.write("Enter some text below to analyze its emotion using our fine-tuned BERTweet + SupCon model!")

# User input box
user_input = st.text_area("Enter your text here:", placeholder="I am feeling wonderful today!")

# Prediction button
if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Tokenize the user's input
            inputs = tokenizer(
                user_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=config["max_length"], 
                padding=False # Matched to your training setup
            )
            
            # Make the prediction
            with torch.no_grad():
                # We only need the logits for prediction, so we ignore the projected features
                logits, _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                prediction_id = torch.argmax(logits, dim=1).item()
            
            # Map the prediction ID to the emotion name and emoji
            emotion_label = config["label2name"][str(prediction_id)]
            emotion_emoji = EMOJI_MAP.get(emotion_label, "✨")
            
            # Display Results
            st.divider()
            st.markdown(f"### Predicted Emotion: **{emotion_label.capitalize()}** {emotion_emoji}")
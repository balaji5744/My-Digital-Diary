import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import pandas as pd
import time
from datetime import datetime
from database import init_db, add_entry, get_all_entries, add_todo, get_todos, update_todo_status

# VADER Import
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Database
init_db()
vader_analyzer = SentimentIntensityAnalyzer()

# --- 1. CONFIGURATION & SETUP ---
with open("config.json", "r") as f:
    config = json.load(f)

# Added "neutral" to the map so the dashboard knows how to display it!
EMOJI_MAP = {
    "sadness": "😢", 
    "joy": "😄", 
    "love": "❤️", 
    "anger": "😡", 
    "fear": "😨", 
    "surprise": "😲",
    "neutral": "😐"
}

SMART_MESSAGES = {
    "sadness": "Hope things get better soon 💙",
    "joy": "Keep smiling! ✨",
    "love": "Cherish these moments ❤️",
    "anger": "Take a deep breath. 🧘",
    "fear": "You are safe. Take it step by step.",
    "surprise": "Expect the unexpected! 😲",
    "neutral": "A calm, balanced day. 🍃"
}

# --- 2. MODEL BLUEPRINT ---
class BertweetSupConClassifier(nn.Module):
    def __init__(self, model_name, num_labels, proj_dim=128, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(cls_embedding)
        projected = F.normalize(self.projection(cls_embedding), p=2, dim=1)
        return logits, projected

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], normalization=True)
    model = BertweetSupConClassifier(config["model_name"], config["num_labels"], config["proj_dim"], config["dropout"])
    model.load_state_dict(torch.load("bertweet_supcon_last_epoch_state_dict.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="My Digital Diary", page_icon="📔", layout="wide")
st.title("📔 My Digital Diary")

tab1, tab2 = st.tabs(["📝 Home & Diary", "📊 Mood Dashboard"])

with tab1:
    # Divide the home screen into left (Diary) and right (Tools)
    left_col, right_col = st.columns([2, 1], gap="large")
    
    with left_col:
        st.header("How are you feeling today?")
        
        quick_emoji = st.selectbox("Quick Emoji Select (Optional):", ["None"] + list(EMOJI_MAP.values()))
        tags = st.text_input("Add tags (e.g., #exam, #friends)")
        user_input = st.text_area("Write your thoughts...", placeholder="I am feeling wonderful today!", height=150)
        
        if st.button("Save Diary Entry", type="primary"):
            if user_input.strip() == "" and quick_emoji == "None":
                st.warning("Please enter some text or select an emoji.")
            else:
                emotion_label = "neutral"
                emotion_emoji = "😐"
                
                if user_input.strip() != "":
                    # --- PREPROCESSING FOR CONTEXT AWARENESS ---
                    # Replace slang/ambiguous words with clear alternatives so the model understands the context
                    import re
                    processed_input = user_input
                    
                    slang_mapping = {
                        r"\bcrush\b": "loved one",
                        r"\bpanipuri\b": "delicious food", # optional: help the model know it's something good
                        # You can add more slang to standard word mappings here
                    }
                    for slang, standard in slang_mapping.items():
                        processed_input = re.sub(slang, standard, processed_input, flags=re.IGNORECASE)

                    # 1. Get prediction from your trained BERTweet model using the processed input
                    inputs = tokenizer(processed_input, return_tensors="pt", truncation=True, max_length=config["max_length"], padding=False)
                    with torch.no_grad():
                        logits, _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                        prediction_id = torch.argmax(logits, dim=1).item()
                    
                    emotion_label = config["label2name"][str(prediction_id)]

                    # --- UPGRADED VADER SAFETY NET ---
                    # Also use the processed input for VADER so it doesn't get confused by the negative definition of "crush"
                    vader_scores = vader_analyzer.polarity_scores(processed_input)
                    compound_score = vader_scores['compound']
                    
                    # Check 1: Is the sentence completely neutral? (Score is very close to 0)
                    if -0.1 < compound_score < 0.1:
                        emotion_label = "neutral"
                        emotion_emoji = "😐"
                    
                    # Check 2: If it's not neutral, is it a negative sentence that the model thought was happy?
                    elif compound_score <= -0.1 and emotion_label in ["joy", "love", "surprise"]:
                        emotion_label = "sadness"
                        emotion_emoji = EMOJI_MAP.get(emotion_label, "😢")
                        
                    else:
                        # If it's not neutral and not a mistake, trust the model!
                        emotion_emoji = EMOJI_MAP.get(emotion_label, "✨")
                    # -----------------------------------
                    
                elif quick_emoji != "None":
                    emotion_label = [k for k, v in EMOJI_MAP.items() if v == quick_emoji][0]
                    emotion_emoji = quick_emoji

                add_entry(user_input, emotion_label, emotion_emoji, tags)
                st.success("Entry Saved!")
                st.markdown(f"### Detected Emotion: **{emotion_label.capitalize()}** {emotion_emoji}")
                st.info(SMART_MESSAGES.get(emotion_label, ""))

    with right_col:
        # --- CALENDAR & FOCUS TIMER WIDGET ---
        st.subheader("📅 Planner & Focus")
        
        # Calendar Picker
        selected_date = st.date_input("Select Date", datetime.now())
        selected_date_str = selected_date.strftime("%Y-%m-%d")
        
        with st.expander("⏱️ Focus Timer", expanded=True):
            t_col1, t_col2 = st.columns([1, 1])
            with t_col1:
                focus_mins = st.number_input("Minutes", min_value=1, max_value=120, value=30, step=5)
            with t_col2:
                st.write("") # Spacing
                st.write("") # Spacing
                if st.button("▶ Focus", use_container_width=True):
                    timer_placeholder = st.empty()
                    for seconds in range(focus_mins * 60, -1, -1):
                        mins, secs = divmod(seconds, 60)
                        timer_placeholder.markdown(f"## ⏳ {mins:02d}:{secs:02d}")
                        time.sleep(1)
                    timer_placeholder.success("Focus session complete!")

        # --- TO-DO LIST WIDGET ---
        st.subheader("📝 To-Do List")
        
        # Add new task
        new_task = st.text_input("Add a new task:", placeholder="Read for 30 mins...")
        if st.button("Add Task"):
            if new_task.strip():
                add_todo(new_task, selected_date_str)
                st.rerun() # Refresh to show new task
                
        # Display existing tasks for the selected date
        todos_df = get_todos(selected_date_str)
        if not todos_df.empty:
            for index, row in todos_df.iterrows():
                # Display a checkbox for each task
                is_checked = st.checkbox(row['task'], value=bool(row['completed']), key=f"todo_{row['id']}")
                # If checkbox state changes, update the database
                if is_checked != bool(row['completed']):
                    update_todo_status(row['id'], is_checked)
                    st.rerun()
        else:
            st.caption("No tasks for this day. Enjoy your free time!")

with tab2:
    st.header("Your Mood Analytics")
    df = get_all_entries()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        most_common = df['emotion'].mode()[0]
        col1.metric("Most Frequent Mood", most_common.capitalize(), EMOJI_MAP.get(most_common, ""))
        col2.metric("Total Entries", len(df))
        
        st.subheader("Emotion Distribution")
        emotion_counts = df['emotion'].value_counts()
        st.bar_chart(emotion_counts)
        
        st.subheader("Filter Your Diary")
        selected_emotion = st.selectbox("Show only entries for:", ["All"] + list(EMOJI_MAP.keys()))
        if selected_emotion != "All":
            filtered_df = df[df['emotion'] == selected_emotion]
            st.dataframe(filtered_df[['date', 'text_content', 'emoji', 'tags']])
        else:
            st.dataframe(df[['date', 'text_content', 'emoji', 'tags']])
    else:
        st.write("No diary entries yet. Go write your first one!")
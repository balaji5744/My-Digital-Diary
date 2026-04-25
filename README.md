📔 My Digital Diary: AI-Powered Mood Tracker & Productivity Hub

My Digital Diary is a full-stack mental wellness and productivity application built with Streamlit. It allows users to log daily journal entries, which are then analyzed by a custom Hybrid AI NLP System to predict the user's core emotion. Alongside mood tracking, the app features an integrated Focus Timer and Daily To-Do list to help users stay productive.

✨ Key Features

🧠 Hybrid Emotion Detection

Deep Learning: Utilizes a custom fine-tuned vinai/bertweet-base model via PyTorch to classify journal entries into 6 core emotions (Joy, Sadness, Anger, Fear, Love, Surprise).

VADER Safety Net: Integrates VADER Sentiment Analysis as a rule-based override to accurately detect Neutral days and correctly handle linguistic negations (e.g., catching "not good" instead of blindly predicting Joy).

📊 Mood Dashboard & Analytics

Emotion Distribution: Visualizes your historical mood data using interactive bar charts.

Quick Filtering: Filter past diary entries by specific emotions to reflect on specific days.

Smart Messaging: Provides dynamic, empathetic UI responses based on your predicted mood.

⏱️ Productivity Suite

Focus Timer: A built-in Pomodoro-style countdown timer (customizable minutes) to keep you on track.

Daily To-Do List: A persistent daily task tracker built directly into the home screen.

💾 Persistent Local Storage

Powered by a lightweight SQLite database to safely store all diary entries, emotion tags, emojis, and daily tasks without losing data between sessions.

🛠️ Tech Stack

Frontend: Streamlit

Machine Learning / NLP: PyTorch, Hugging Face Transformers, VADER Sentiment Analysis

Data Handling: Pandas, SQLite3

Language: Python 3.x

📂 Project Structure

My-Digital-Diary/

│

├── app.py                   # Main Streamlit application and UI routing

├── database.py              # SQLite database initialization and query functions

├── config.json              # Model configuration and label mapping

├── requirements.txt         # Project dependencies

├── bertweet_supcon...pth    # Fine-tuned PyTorch model weights (Not in repo due to size limit)

└── .gitignore               # Ignored files

🚀 Installation & Setup

To run this application locally, follow these steps:

1. Clone the repository

git clone https://github.com/your-username/my-digital-diary.git

cd my-digital-diary

2. Create a virtual environment (Recommended)

python -m venv venv

source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies

pip install -r requirements.txt

4. Add the Model Weights

Note: Due to GitHub file size limits, the fine-tuned .pth model file is not included in this repository.

Place your trained PyTorch weights file (e.g., bertweet_supcon_last_epoch_state_dict.pth) in the root directory.

5. Run the Application

streamlit run app.py

The app will open automatically in your browser at http://localhost:8501.

💡 How the AI Works (The Ensemble Approach)

Language models often struggle with short text negations. If a user writes "I am not feeling good", a standard ML model might focus heavily on the word "good" and misclassify the text as Joy.

To solve this, this app uses an Ensemble/Hybrid approach:

The BERTweet model generates an initial emotional prediction.

The VADER Sentiment Analyzer calculates the compound polarity score of the text.

The Override: If VADER detects a highly negative polarity but the model predicted a positive emotion, the app dynamically overrides the prediction to Sadness/Anger. If VADER detects a polarity close to 0.0, the app logs the entry as Neutral.

🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Built with ❤️ and Python.


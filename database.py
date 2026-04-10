import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "diary.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Diary Entries Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            text_content TEXT,
            emotion TEXT,
            emoji TEXT,
            tags TEXT
        )
    ''')
    # NEW: To-Do List Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            task TEXT,
            completed BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# --- Diary Functions ---
def add_entry(text, emotion, emoji, tags):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d")
    c.execute('INSERT INTO entries (date, text_content, emotion, emoji, tags) VALUES (?, ?, ?, ?, ?)', 
              (date_str, text, emotion, emoji, tags))
    conn.commit()
    conn.close()

def get_all_entries():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM entries", conn)
    conn.close()
    return df

# --- NEW: To-Do Functions ---
def add_todo(task, date_str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO todos (date, task, completed) VALUES (?, ?, ?)', (date_str, task, False))
    conn.commit()
    conn.close()

def get_todos(date_str):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM todos WHERE date = ?", conn, params=(date_str,))
    conn.close()
    return df

def update_todo_status(task_id, completed):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE todos SET completed = ? WHERE id = ?', (completed, task_id))
    conn.commit()
    conn.close()
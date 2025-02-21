import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from textblob import TextBlob
import time
import random
import matplotlib.pyplot as plt
import json
import hashlib
import os
from collections import Counter

# Function to detect facial emotion
def detect_facial_emotion(frame):
    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    if result:
        return result[0]['dominant_emotion'].capitalize()
    return "Neutral"

# Function to detect text emotion
def detect_text_emotion(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Happy"
    elif polarity < 0:
        return "Sad"
    return "Fear"

# Function to monitor stress levels
def monitor_stress(mood_history):
    stress_threshold = 3
    recent_moods = [entry["mood"] for entry in mood_history[-5:] if isinstance(entry, dict) and "mood" in entry]
    
    stressed_count = sum(1 for mood in recent_moods if mood == "Sad")

    if stressed_count >= stress_threshold:
        print("⚠️ ALERT: Prolonged stress detected! Notifying HR.")

# Function to recommend tasks based on mood
def recommend_task(mood):
    mood = mood.capitalize()
    tasks = {
        "Happy": ["Collaborate on a new project", "Share positivity with the team"],
        "Sad": ["Take a break", "Listen to music", "Talk to a friend"],
        "Fear": ["Practice deep breathing", "Engage in light work", "Seek support"],
        "Angry": ["Cool down with a short walk", "Work on solo tasks"],
        "Surprise": ["Reflect on new insights", "Plan next steps"]
    }
    return random.choice(tasks.get(mood, ["No suggestion available"]))

# Function to anonymize mood data (but store actual mood separately)
def anonymize_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

# Function to save mood history properly
# Function to save only employee_id and anonymized mood
def save_mood_history(mood_history, filename="mood_history.json"):
    try:
        filtered_history = [
            {"employee_id": entry["employee_id"], "anonymized_mood": entry["anonymized_mood"]}
            for entry in mood_history
        ]

        with open(filename, "w") as file:
            json.dump(filtered_history, file, indent=4)  

        print("✅ Mood history successfully updated with anonymized data!")
    except Exception as e:
        print(f"❌ Error saving mood history: {e}")


# Function to load mood history properly
def load_mood_history(filename="mood_history.json"):
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        return []  
    
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("⚠️ Warning: mood_history.json is corrupted or empty. Resetting...")
        return []

# Function to plot team mood analysis
def plot_team_mood(mood_history):
    if not mood_history:
        print("⚠️ No mood history available to plot.")
        return
    
    mood_counts = Counter(entry["mood"] for entry in mood_history if isinstance(entry, dict) and "mood" in entry)
    
    if not mood_counts:  # Prevent errors if no valid moods exist
        print("⚠️ No valid mood data available for plotting.")
        return

    colors = {
        "Happy": "green",
        "Sad": "black",
        "Fear": "red",
        "Surprise": "orange",
        "Angry": "blue",
        "Neutral": "gold"
    }

    moods, counts = zip(*mood_counts.items())
    bar_colors = [colors.get(mood, "gray") for mood in moods]

    plt.bar(moods, counts, color=bar_colors)
    plt.xlabel("Mood")
    plt.ylabel("Count")
    plt.title("Team Mood Analysis")
    plt.show()

# Main function
def main():
    cap = cv2.VideoCapture(0)
    mood_history = load_mood_history()
    
    employee_id = input("Enter Employee ID: ").strip()  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detected_mood = detect_facial_emotion(frame)
        
        mood_entry = {
            "employee_id": employee_id,
            "mood": detected_mood,  
            "anonymized_mood": anonymize_data(detected_mood)
        }
        
        mood_history.append(mood_entry)  
        save_mood_history(mood_history)  
        monitor_stress(mood_history)
        
        print(f"🧐 Employee: {employee_id} | Detected Mood: {detected_mood}")
        print(f"🎯 Recommended Task: {recommend_task(detected_mood)}")
        
        cv2.imshow('Real-Time Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    plot_team_mood(mood_history)

if __name__ == "__main__":
    main()

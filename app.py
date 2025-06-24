from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
from collections import Counter
from googleapiclient.discovery import build
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# === Load Trained Model and Vectorizer ===
model = joblib.load("./artifacts/text_classifier_model.pkl")
vectorizer = joblib.load("./artifacts/text_preprocessor.pkl")

# === YouTube API Setup ===
load_dotenv()  # Load variables from .env

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

# === Fetch YouTube Comments ===
def get_comments(video_id, max_results=100):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText"
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

@app.route("/")
def home():
    return "YouTube Sentiment Analysis API is running locally."

@app.route("/analyze", methods=["GET"])
def analyze_video():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "Missing video_id"}), 400

    comments = get_comments(video_id)
    if not comments:
        return jsonify({"positive": 0, "negative": 0, "neutral": 0, "comments": []})

    try:
        X = vectorizer.transform(comments)
        preds = model.predict(X)
        counts = Counter(preds)

        return jsonify({
            "positive": int(counts.get(1, 0)),
            "neutral": int(counts.get(0, 0)),
            "negative": int(counts.get(-1, 0)),
            "comments": comments
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    try:
        data = request.get_json()
        comments = data.get("comments", [])
        if not comments:
            return jsonify([])

        texts = [c["text"] for c in comments]
        timestamps = [c["timestamp"] for c in comments]
        X = vectorizer.transform(texts)
        preds = model.predict(X)

        results = []
        for text, sentiment, ts in zip(texts, preds, timestamps):
            results.append({
                "comment": text,
                "sentiment": int(sentiment),
                "timestamp": ts
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get("sentiment_counts", {})
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [sentiment_counts.get("1", 0), sentiment_counts.get("0", 0), sentiment_counts.get("-1", 0)]
        colors = ['green', 'gray', 'red']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.axis('equal')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get("comments", [])
        text = " ".join(comments)

        wc = WordCloud(width=600, height=400, background_color='black').generate(text)

        buf = BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get("sentiment_data", [])
        if not sentiment_data:
            return jsonify({"error": "No sentiment data"}), 400

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)

        plt.figure(figsize=(10, 4))
        plt.plot(df["timestamp"], df["sentiment"], marker='o', linestyle='-')
        plt.xlabel("Time")
        plt.ylabel("Sentiment")
        plt.grid(True)
        plt.title("Sentiment Trend Over Time")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

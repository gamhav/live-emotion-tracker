# 🎭 Emotion Tracker

A live camera-based emotion detector with a sci-fi facial grid overlay.
Built with Python Flask + OpenCV. Clean, minimal UI.

## Stack
- **Backend**: Python 3 + Flask + OpenCV
- **Frontend**: Vanilla HTML/CSS/JS
- **Detection**: Haar Cascades (eye, smile, face) + heuristic scoring

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser
# → http://127.0.0.1:5000
```

## How it works

1. Flask streams MJPEG frames from your webcam via `/video_feed`
2. Each frame is analysed using OpenCV Haar Cascades:
   - Face detection
   - Eye detection (openness, count)
   - Smile detection
   - Mouth region variance (open/closed)
3. A heuristic scoring engine maps these signals to 7 emotions:
   `Happy · Sad · Angry · Surprised · Neutral · Fearful · Disgusted`
4. The UI polls `/emotion_data` every 800ms for live updates

## Features
- 🔲 Sci-fi wireframe grid overlay on detected face
- 📊 Live emotion breakdown bar chart
- 🕒 Session emotion log
- 🟢 Animated scan line on camera feed
- Corner bracket markers on face region
- Color-coded per emotion

## Project Structure
```
emotion_tracker/
├── app.py               ← Flask server + detection logic
├── templates/
│   └── index.html       ← Frontend UI
├── requirements.txt
└── README.md
```

## Upgrade ideas
- Add `deepface` or `fer` for ML-based detection (more accurate)
- Save emotion history to SQLite
- Export session report as CSV
- Add sound alerts on emotion change

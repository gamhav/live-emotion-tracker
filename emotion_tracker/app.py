from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import random
import time
import threading
import json

app = Flask(__name__)

# ─── Emotion Engine ───────────────────────────────────────────────────────────
# Uses facial geometry heuristics (eye openness, mouth shape, brow position)
# to estimate emotion — no external ML model needed.

EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Fearful", "Disgusted"]
EMOTION_COLORS = {
    "Happy":     (50,  220, 120),
    "Sad":       (90,  130, 220),
    "Angry":     (50,   60, 220),
    "Surprised": (50,  210, 240),
    "Neutral":   (160, 160, 160),
    "Fearful":   (180,  80, 220),
    "Disgusted": (60,  180,  80),
}

face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

current_emotion = {"label": "Scanning...", "confidence": 0, "scores": {}}
lock = threading.Lock()


def estimate_emotion(face_roi_gray, face_roi_color):
    """Heuristic emotion estimator using face sub-features."""
    h, w = face_roi_gray.shape
    scores = {e: random.uniform(2, 10) for e in EMOTIONS}  # base noise

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(20, 20))
    eye_count = len(eyes)

    # Detect smile
    smiles = smile_cascade.detectMultiScale(
        face_roi_gray, 1.7, 22,
        minSize=(int(w * 0.3), int(h * 0.1)),
        maxSize=(int(w * 0.9), int(h * 0.4))
    )
    smile_detected = len(smiles) > 0

    # Basic brightness variance in lower half (mouth region) → open mouth
    lower = face_roi_gray[h // 2:, :]
    mouth_variance = float(np.std(lower))

    # Eye openness: mean brightness of eye region
    upper = face_roi_gray[:h // 2, :]
    eye_brightness = float(np.mean(upper))

    # Heuristic scoring
    if smile_detected:
        scores["Happy"] += 55
        scores["Neutral"] -= 10
    if eye_count >= 2:
        if mouth_variance > 30:
            scores["Surprised"] += 35
            scores["Fearful"] += 15
        if eye_brightness > 130:
            scores["Surprised"] += 10
    if mouth_variance > 40:
        scores["Angry"] += 20
        scores["Disgusted"] += 15
    if not smile_detected and mouth_variance < 20:
        scores["Neutral"] += 30
        scores["Sad"] += 15
    if eye_count < 2 and not smile_detected:
        scores["Sad"] += 25
        scores["Neutral"] += 10

    # Normalise to percentages
    total = sum(scores.values())
    scores = {k: round((v / total) * 100, 1) for k, v in scores.items()}
    top = max(scores, key=scores.get)
    return top, scores[top], scores


def draw_grid_overlay(frame, x, y, w, h, color, n=8):
    """Draw a cool wireframe grid over the face region."""
    alpha_frame = frame.copy()
    step_x = w // n
    step_y = h // n
    c = color

    # Vertical lines
    for i in range(n + 1):
        lx = x + i * step_x
        cv2.line(alpha_frame, (lx, y), (lx, y + h), c, 1)
    # Horizontal lines
    for i in range(n + 1):
        ly = y + i * step_y
        cv2.line(alpha_frame, (x, ly), (x + w, ly), c, 1)
    # Diagonal cross lines for sci-fi look
    cv2.line(alpha_frame, (x, y), (x + w, y + h), c, 1)
    cv2.line(alpha_frame, (x + w, y), (x, y + h), c, 1)
    # Corner brackets
    blen = w // 6
    thick = 2
    for bx, by, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(frame, (bx, by), (bx + dx * blen, by), c, thick)
        cv2.line(frame, (bx, by), (bx, by + dy * blen), c, thick)

    cv2.addWeighted(alpha_frame, 0.35, frame, 0.65, 0, frame)

    # Solid bright border
    cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)


def draw_hud(frame, emotion, confidence, scores):
    """Minimal HUD overlay."""
    h_f, w_f = frame.shape[:2]
    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w_f, 38), (10, 10, 10), -1)
    cv2.putText(frame, "EMOTION TRACKER  //  LIVE", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (w_f - 90, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)

    # Bottom emotion label
    label = f"{emotion}  {confidence:.0f}%"
    cv2.rectangle(frame, (0, h_f - 44), (w_f, h_f), (10, 10, 10), -1)
    cv2.putText(frame, label, (12, h_f - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mini bar chart on right
    bar_x = w_f - 160
    bar_y = 50
    bar_w = 140
    bar_h = 14
    gap = 4
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:5]
    for i, (emo, score) in enumerate(sorted_scores):
        by = bar_y + i * (bar_h + gap)
        filled = int((score / 100) * bar_w)
        ec = EMOTION_COLORS.get(emo, (180, 180, 180))
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (30, 30, 30), -1)
        cv2.rectangle(frame, (bar_x, by), (bar_x + filled, by + bar_h), ec, -1)
        cv2.putText(frame, f"{emo[:7]:7s} {score:.0f}%", (bar_x, by + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def generate_frames():
    global current_emotion
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    last_emotion = "Neutral"
    last_confidence = 0
    last_scores = {e: 0 for e in EMOTIONS}

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            # Pick largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_gray  = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            # Re-analyse every 6 frames for perf
            if frame_count % 6 == 0:
                emotion, conf, scores = estimate_emotion(face_gray, face_color)
                last_emotion, last_confidence, last_scores = emotion, conf, scores
                with lock:
                    current_emotion = {"label": emotion, "confidence": conf, "scores": scores}

            color = EMOTION_COLORS.get(last_emotion, (200, 200, 200))
            draw_grid_overlay(frame, x, y, w, h, color)
            draw_hud(frame, last_emotion, last_confidence, last_scores)
        else:
            # No face
            cv2.putText(frame, "NO FACE DETECTED", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
            draw_hud(frame, "Scanning...", 0, {e: 0 for e in EMOTIONS})
            with lock:
                current_emotion = {"label": "Scanning...", "confidence": 0, "scores": {}}

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/emotion_data")
def emotion_data():
    with lock:
        return jsonify(current_emotion)

if __name__ == "__main__":
    print("\n🎭  Emotion Tracker running at  →  http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True)

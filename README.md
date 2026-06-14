# 👁 FaceTrack — Smart Attendance System

A full-stack face recognition attendance system built with **Python + OpenCV + Flask**.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5050
```

---

## 📁 Project Structure

```
attendance_system/
├── app.py                  # Flask backend + REST API
├── requirements.txt
├── database/
│   └── db.py               # SQLite database layer
├── models/
│   └── face_engine.py      # OpenCV LBPH face recognition
├── templates/
│   └── index.html          # Full SPA frontend
└── uploads/                # Member photos stored here
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Face Recognition** | OpenCV LBPH recognizer — works offline, no API keys needed |
| **SQLite Database** | All attendance data persisted locally |
| **Add Members** | Upload a photo to register new person |
| **Check In / Out** | Upload face photo → auto-marks attendance |
| **Dashboard** | Real-time today's attendance log |
| **Analytics** | Attendance % per person, 7-day trend chart |
| **History** | Full attendance records with filtering |

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/members` | List all members |
| POST | `/api/members` | Add new member |
| DELETE | `/api/members/<id>` | Remove member |
| POST | `/api/recognize` | Recognize face + mark attendance |
| GET | `/api/attendance/today` | Today's attendance |
| GET | `/api/attendance/history?days=30` | Historical records |
| GET | `/api/stats` | Attendance statistics |
| POST | `/api/train` | Retrain face model |

---

## 📸 How Face Recognition Works

1. **Training**: OpenCV `LBPHFaceRecognizer` trained on member photos
2. **Detection**: Haar Cascade classifier detects faces in uploaded images
3. **Recognition**: LBPH compares histograms; confidence ≥ 50% = recognized
4. **Auto-retrain**: Model retrains automatically when a new member is added

---

## 💡 Tips for Best Accuracy

- Use **clear, well-lit frontal face photos** when registering
- Avoid sunglasses or heavy shadows
- Upload 1 good quality photo per person
- For better accuracy, you can upload multiple photos and retrain

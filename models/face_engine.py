import cv2
import numpy as np
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.db import get_all_members

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_model.yml')
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'labels.pkl')

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_faces(image_gray):
    """Detect faces in a grayscale image."""
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def preprocess_face(image_gray, x, y, w, h):
    """Extract and preprocess a face ROI."""
    face_roi = image_gray[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (200, 200))
    face_roi = cv2.equalizeHist(face_roi)
    return face_roi

def extract_faces_from_image(img_path):
    """Extract face ROIs from an image file. Returns list of (face_roi, x,y,w,h)."""
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    result = []
    for (x, y, w, h) in faces:
        roi = preprocess_face(gray, x, y, w, h)
        result.append((roi, x, y, w, h))
    return result

def extract_faces_from_bytes(img_bytes):
    """Extract face ROIs from image bytes."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    result = []
    for (x, y, w, h) in faces:
        roi = preprocess_face(gray, x, y, w, h)
        result.append((roi, x, y, w, h))
    return img, result

def train_model():
    """Train the LBPH model from all member photos and save to DB."""
    import base64
    import pickle
    from database.db import get_all_members, save_model_to_db

    members = get_all_members()
    faces_data = []
    labels_data = []

    for member in members:
        photo_data = member.get('photo_data', '')
        if not photo_data:
            continue
        try:
            photo_bytes = base64.b64decode(photo_data)
        except Exception:
            continue
        img, face_data = extract_faces_from_bytes(photo_bytes)
        if face_data:
            roi, x, y, w, h = face_data[0]
            faces_data.append(roi)
            labels_data.append(member['label'])

    if len(faces_data) < 1:
        return False, "No training data available. Add members with photos first."

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    recognizer.train(faces_data, np.array(labels_data))

    # Save to temp file then read as bytes → store in DB
    tmp_path = '/tmp/face_model_tmp.yml'
    recognizer.save(tmp_path)
    with open(tmp_path, 'rb') as f:
        model_bytes = f.read()

    labels_bytes = pickle.dumps({m['label']: m['name'] for m in members})
    save_model_to_db(model_bytes, labels_bytes)

    return True, f"Model trained with {len(faces_data)} faces."

def load_recognizer():
    """Load the trained recognizer."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        return recognizer
    except Exception:
        return None

def recognize_face(img_bytes, threshold=80):
    """
    Recognize face(s) in image bytes.
    Returns list of dicts with label, confidence, bbox.
    """
    recognizer,labels = load_recognizer()
    if recognizer is None:
        return None, "Model not trained yet. Please add members and train."

    img, face_data = extract_faces_from_bytes(img_bytes)
    if img is None:
        return None, "Could not read image."
    if not face_data:
        return [], "No face detected in image."

    results = []
    for (roi, x, y, w, h) in face_data:
        label, confidence = recognizer.predict(roi)
        # LBPH confidence: lower = more similar. Convert to similarity score
        similarity = max(0, 100 - confidence)
        recognized = similarity >= (100 - threshold)
        results.append({
            'label': label if recognized else -1,
            'confidence': round(similarity, 1),
            'raw_confidence': round(confidence, 1),
            'recognized': recognized,
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })

    return results, None

def get_annotated_image(img_bytes, results, member_name=None):
    """Draw bounding boxes and labels on image. Returns JPEG bytes."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    for r in results:
        bb = r['bbox']
        x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']

        if r['recognized']:
            color = (0, 200, 80)
            label_text = f"{member_name or 'Known'} ({r['confidence']:.0f}%)"
        else:
            color = (0, 80, 220)
            label_text = f"Unknown ({r['confidence']:.0f}%)"

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

        # Label background
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x, y - text_h - 14), (x + text_w + 10, y), color, -1)
        cv2.putText(img, label_text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()

def model_exists():
    return os.path.exists(MODEL_PATH)

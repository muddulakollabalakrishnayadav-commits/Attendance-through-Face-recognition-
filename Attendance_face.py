import cv2
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

# ----- SETTINGS -----
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance_logs"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.json"
CAMERA_INDEX = 0

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----- 1. CAPTURE IMAGES FOR A PERSON -----
def capture_images(name):
    cam = cv2.VideoCapture(CAMERA_INDEX)
    count = 0
    user_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    print(f"\n[INFO] Capturing face images for: {name}")
    print("[INFO] Look at the camera. Press 'q' to stop early.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(user_dir, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if count >= 30:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} images for {name} in {user_dir}")


# ----- 2. TRAIN MODEL FROM DATASET -----
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}  # id -> name
    current_label = 0

    print("\n[INFO] Starting training...")

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Reading images for: {person_name}")
        label_map[current_label] = person_name

        for img_file in os.listdir(person_path):
            if not (img_file.lower().endswith(".jpg") or img_file.lower().endswith(".png")):
                continue

            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    if len(faces) == 0:
        print("[ERROR] No images found in dataset/. Add people with option 1 first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)

    # Save label map as JSON
    with open(LABELS_FILE, "w") as f:
        json.dump(label_map, f)

    print(f"[INFO] Training complete.")
    print(f"[INFO] Saved model to {TRAINER_FILE}")
    print(f"[INFO] Saved labels to {LABELS_FILE}")


# ----- ATTENDANCE HELPER -----
def mark_attendance(name):
    today_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    file_path = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")

    # Create or load existing file
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Avoid duplicate entries for same person on same date
    already_marked = ((df["Name"] == name) & (df["Date"] == today_str)).any()
    if already_marked:
        return

    df.loc[len(df)] = [name, today_str, time_str]
    df.to_csv(file_path, index=False)
    print(f"[MARKED] {name} at {today_str} {time_str}")


# ----- 3. RECOGNIZE FACES & TAKE ATTENDANCE -----
def recognize_faces():
    if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
        print("[ERROR] Model not trained yet. Run option 2 first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    # Load labels
    with open(LABELS_FILE, "r") as f:
        label_map = json.load(f)
    # keys come as strings from json, convert to int
    label_map = {int(k): v for k, v in label_map.items()}

    cam = cv2.VideoCapture(CAMERA_INDEX)
    print("\n[INFO] Starting attendance. Press 'q' to stop.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            id_, conf = recognizer.predict(face_img)
            # Lower conf = better match; adjust threshold if needed
            if conf < 70:
                name = label_map.get(id_, "Unknown")
                mark_attendance(name)
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Attendance session ended.")


# ----- MENU -----
def main():
    print("\n===== FACE ATTENDANCE SYSTEM =====")
    print("1. Add New Person (capture images)")
    print("2. Train Model")
    print("3. Start Attendance")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        name = input("Enter person name: ").strip()
        if name == "":
            print("[ERROR] Name cannot be empty.")
        else:
            capture_images(name)
    elif choice == "2":
        train_model()
    elif choice == "3":
        recognize_faces()
    else:
        print("[ERROR] Invalid choice.")


if __name__ == "__main__":
    main()
    

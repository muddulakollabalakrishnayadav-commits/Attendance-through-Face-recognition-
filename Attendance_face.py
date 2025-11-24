import os
from datetime import datetime

import pandas as pd
from deepface import DeepFace

KNOWN_DIR = "/storage/emulated/0/face_attendance/known_faces"
ATTENDANCE_FILE = "/storage/emulated/0/face_attendance/attendance.csv"

def init_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

def list_known_faces():
    files = []
    for f in os.listdir(KNOWN_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(f)
    return files

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    if name in df["Name"].values:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat(
        [df, pd.DataFrame([{"Name": name, "Time": now}])],
        ignore_index=True
    )
    df.to_csv(ATTENDANCE_FILE, index=False)

def process_image(captured_image_path, model_name="VGG-Face"):
    init_attendance_file()
    known_files = list_known_faces()
    if not known_files:
        print("No known faces in", KNOWN_DIR)
        return

    found_names = set()

    for f in known_files:
        known_path = os.path.join(KNOWN_DIR, f)
        name = os.path.splitext(f)[0]

        try:
            result = DeepFace.verify(
                img1_path=known_path,
                img2_path=captured_image_path,
                model_name=model_name,
                enforce_detection=False
            )
            if result.get("verified"):
                found_names.add(name)
        except Exception as e:
            print("Error with", name, ":", e)

    if not found_names:
        print("No known faces detected.")
    else:
        for name in found_names:
            mark_attendance(name)
            print("Marked present:", name)

if __name__ == "__main__":
    # Change this per photo or accept from command line
    captured = "/storage/emulated/0/face_attendance/captured/class1.jpg"
    process_image(captured)

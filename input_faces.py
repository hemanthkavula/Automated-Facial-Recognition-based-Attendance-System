import cv2
import pickle
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path('data')
FACE_CASCADE_PATH = str(DATA_DIR / 'haarcascade_frontalface_default.xml')
IMG_SIZE = (50, 50)
SAMPLES_PER_PERSON = 100

DATA_DIR.mkdir(exist_ok=True)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(FACE_CASCADE_PATH)

faces_data = []
sample_count = 0

name = input("Enter Your Name: ").strip()
if not name:
    print("Name cannot be empty. Exiting...")
    video.release()
    exit()

print(f"Collecting face data for: {name}")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error accessing the camera. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, IMG_SIZE)
        if sample_count % 10 == 0 and len(faces_data) < SAMPLES_PER_PERSON:
            faces_data.append(resized_img)

        sample_count += 1
        cv2.putText(frame, f"Samples: {len(faces_data)}/{SAMPLES_PER_PERSON}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= SAMPLES_PER_PERSON:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(SAMPLES_PER_PERSON, -1)

names_file = DATA_DIR / 'names.pkl'
faces_file = DATA_DIR / 'faces_data.pkl'

if not names_file.exists():
    names = [name] * SAMPLES_PER_PERSON
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * SAMPLES_PER_PERSON)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

if not faces_file.exists():
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print(f"Face data collection complete for {name}! {SAMPLES_PER_PERSON} samples saved.")
print(f"Total dataset size: {faces.shape[0]} samples.")

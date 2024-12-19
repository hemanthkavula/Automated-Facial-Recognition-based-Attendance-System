import cv2
import pickle
import numpy as np
import os
import csv
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths and Directories
DATA_DIR = Path('data')
ATTENDANCE_DIR = Path('Attendance')
FACE_CASCADE_PATH = str(DATA_DIR / 'haarcascade_frontalface_default.xml')
IMG_SIZE = (50, 50)

DATA_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)

# Load Data
try:
    with open(DATA_DIR / 'names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open(DATA_DIR / 'faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    print(f"Model loaded successfully. Dataset size: {FACES.shape}")
except FileNotFoundError:
    print("Model data not found. Ensure training is completed.")
    exit()

# Normalize Data
scaler = StandardScaler()
FACES = scaler.fit_transform(FACES)

# Reduce Training Data (use only 20% of data to reduce accuracy)
FACES = FACES[:int(len(FACES) * 0.2)]
LABELS = LABELS[:int(len(LABELS) * 0.2)]

# Add more noise to the data to decrease accuracy
noise_factor = 0.8  # Increased noise to make it harder for the model
FACES += noise_factor * np.random.randn(*FACES.shape)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.3, random_state=42)

# Train KNN Model with a higher value for neighbors to reduce performance
knn = KNeighborsClassifier(n_neighbors=30, weights='distance')  # Try a larger value and 'distance' weight
knn.fit(X_train, y_train)

# Evaluate Model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# ---- Visualization ----

# 1. Accuracy vs. Number of Neighbors
neighbors = list(range(1, 31))
accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(neighbors, accuracies)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors')
plt.show()


# 3. Training vs. Testing Accuracy with different train-test splits
train_accuracies = []
test_accuracies = []

for test_size in np.linspace(0.1, 0.5, 5):
    X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=test_size, random_state=42)
    knn.fit(X_train, y_train)
    
    train_accuracies.append(knn.score(X_train, y_train))
    test_accuracies.append(knn.score(X_test, y_test))

plt.plot(np.linspace(0.1, 0.5, 5), train_accuracies, label='Train Accuracy')
plt.plot(np.linspace(0.1, 0.5, 5), test_accuracies, label='Test Accuracy')
plt.xlabel('Test Size')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy vs Test Size')
plt.legend()
plt.show()

# Real-Time Attendance System
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error accessing the camera. Exiting...")
    exit()

COL_NAMES = ['NAME', 'TIME', 'CONFIDENCE']
facedetect = cv2.CascadeClassifier(FACE_CASCADE_PATH)

while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Error capturing video frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, IMG_SIZE).flatten().reshape(1, -1)
            resized_img = scaler.transform(resized_img)  # Normalize real-time input

            # Predict the class and calculate confidence
            distances, indices = knn.kneighbors(resized_img)
            prediction = knn.predict(resized_img)[0]
            confidence = 1 - (distances[0][0] / distances[0].sum())  # Confidence as a normalized distance
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            date = datetime.now().strftime("%Y-%m-%d")
            file_path = ATTENDANCE_DIR / f"Attendance_{date}.csv"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
            text = f"{prediction} ({accuracy:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Record attendance and stop system after saving
            attendance = [prediction, timestamp, f"{accuracy:.2f}"]
            if cv2.waitKey(1) & 0xFF == ord('o'):
                if not file_path.exists():
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
                
                # Stop the system after recording attendance
                print(f"Attendance recorded for {prediction} at {timestamp}")
                break  # Exit the loop after recording attendance
            
        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
print("Attendance system closed.")

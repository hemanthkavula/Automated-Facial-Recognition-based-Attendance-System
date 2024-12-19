import cv2
import pickle
import numpy as np
import os
import csv
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path('data')
ATTENDANCE_DIR = Path('Attendance')
FACE_CASCADE_PATH = str(DATA_DIR / 'haarcascade_frontalface_default.xml')
IMG_SIZE = (50, 50)

DATA_DIR.mkdir(exist_ok=True)
ATTENDANCE_DIR.mkdir(exist_ok=True)


try:
    with open(DATA_DIR / 'names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open(DATA_DIR / 'faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    print(f"Model loaded successfully. Dataset size: {FACES.shape}")
except FileNotFoundError:
    print("Model data not found. Ensure training is completed.")
    exit()


scaler = MinMaxScaler()
FACES = scaler.fit_transform(FACES)

pca = PCA(n_components=50) 
FACES = pca.fit_transform(FACES)

X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform') 
knn.fit(X_train, y_train)

neighbors = list(range(1, 100))
accuracies = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform') 
    cv_scores = cross_val_score(knn, FACES, LABELS, cv=5)
    accuracies.append(cv_scores.mean()) 

plt.plot(neighbors, accuracies)
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors (Cross-Validation)')
plt.show()

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

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error accessing the camera. Exiting...")
    exit()

COL_NAMES = ['NAME', 'TIME', 'ACCURACY']
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
            resized_img = scaler.transform(resized_img) 

            resized_img = pca.transform(resized_img)

            distances, indices = knn.kneighbors(resized_img)
            prediction = knn.predict(resized_img)[0]
            confidence = 1 - (distances[0][0] / distances[0].sum()) 

            timestamp = datetime.now().strftime("%H:%M:%S")
            date = datetime.now().strftime("%Y-%m-%d")
            file_path = ATTENDANCE_DIR / f"Attendance_{date}.csv"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
            text = f"{prediction} ({accuracy:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            attendance = [prediction, timestamp, f"{accuracy:.2f}"]
            if cv2.waitKey(1) & 0xFF == ord('o'):
                if not file_path.exists():
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)

                print(f"Attendance recorded for {prediction} at {timestamp}")
                break 

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

video.release()
cv2.destroyAllWindows()
print("Attendance system closed.")

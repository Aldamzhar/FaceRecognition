import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def autolabel(rectangulars):
    for rectangular in rectangulars:
        height = rectangular.get_height()
        width = rectangular.get_width()
        x = rectangular.get_x()
        ax.annotate(
            "{:.3f}".format(height),
            xy=(x+width / 2, height),
            xytext=(0, 3), 
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

def load_and_preprocess_face(path):
    img = cv2.imread(path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    img = preprocess_input(keras_image.img_to_array(img))
    return img

def compute_embedding(model, face):
    return model.predict(np.expand_dims(face, axis=0))
    
true_positives = 0
false_positives = 0
false_negatives = 0
total_faces = 0

path = "real_and_fake_face_detection/real_and_fake_face/training_real_/"
face_db = {}
threshold = 0.6

for file in os.listdir(path):
    name = os.path.splitext(file)[0]
    face = load_and_preprocess_face(os.path.join(path, file))
    face_db[name] = compute_embedding(model, face)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    faces = classifier.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5)

    for (x, y, width, height) in faces:
        face_img = cv2.resize(frame[y:y+height, x:x+width], (224, 224))
        face_img = preprocess_input(keras_image.img_to_array(face_img))
        face_embedding = compute_embedding(model, face_img)

        recognized_name = None
        min_distance = float('inf')
        for name, db_embedding in face_db.items():
            distance = cosine(face_embedding[0], db_embedding[0])
            if distance < min_distance and distance < threshold:
                min_distance = distance
                recognized_name = name

        total_faces += 1

        if name == recognized_name:
            true_positives += 1
        elif recognized_name is not None:
            false_positives += 1
        else:
            false_negatives += 1        
                
        if recognized_name:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break        

video_capture.release()
cv2.destroyAllWindows()

accuracy = true_positives / total_faces
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

metrics = [accuracy, precision, recall, f1_score]

labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects = ax.bar(x, metrics, width=0.4)
autolabel(rects)

ax.set_ylabel("Score")
ax.set_title("Evaluation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.show()





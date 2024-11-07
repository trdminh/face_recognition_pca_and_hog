import cv2

from skimage.feature import hog
from sklearn.decomposition import PCA

import pickle as pk
import joblib
import numpy as np

clf = joblib.load('c2_svm_classifier.pkl')
pca = pk.load(open("c1_PCA.pkl",'rb'))
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)        
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]  # Cắt khuôn mặt
        resized = cv2.resize(face_roi, (64, 128))
        hog_features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        
        hog_features_pca = pca.transform([hog_features])  
        probabilities = clf.predict_proba(hog_features_pca)  
        confidence = np.max(probabilities)
        label = clf.predict(hog_features_pca)[0]

        if confidence < 0.9:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            if (label == 0):
                cv2.putText(frame, f'Ha', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Face Recognition with Saved Model', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
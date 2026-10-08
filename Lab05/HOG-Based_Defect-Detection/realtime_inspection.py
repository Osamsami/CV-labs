import time
import cv2
import joblib
import numpy as np
from skimage.feature import hog

bundle = joblib.load('hog_svm_model.joblib')
model, CELL, ORI, IMG = bundle['model'], bundle['cell'], bundle['ori'], bundle['img']

def predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(gray[y:y + s, x:x + s], (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    f = hog(crop, orientations=ORI, pixels_per_cell=(CELL, CELL),
            cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
    prob = model.predict_proba(f)[0]
    pred = int(prob[1] >= 0.5)
    return pred, prob[pred] * 100, (x, y, s)

cap = cv2.VideoCapture(0)
prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    pred, conf, (x, y, s) = predict(frame)
    label = 'DEFECTIVE' if pred else 'PASS'
    color = (0, 0, 255) if pred else (0, 200, 0)
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.rectangle(frame, (x, y), (x + s - 1, y + s - 1), color, 4)
    cv2.putText(frame, f'{label} {conf:.0f}%', (x + 10, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f'FPS {fps:.1f}', (x + 10, y + s - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Lab05 Inspection (press q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 200)

while True:
    value, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        a = gray[y:y + h, x:x + w]
        b = img[y:y + h, x:x + w]

    cv2.imshow('video', img)
    if cv2.waitKey(1) and 0xff ==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


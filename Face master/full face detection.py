import numpy as np
import cv2

#@code by abdisa merga 9/2/2020
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
cam.set(10,200)

while True:
    try:

        value, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            a = gray[y:y + h, x:x + w]
            b = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(a,scaleFactor=1.5,minNeighbors=5,minSize=(5, 5),)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(b, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            smile = smile_cascade.detectMultiScale(a,scaleFactor=1.5,minNeighbors=15,minSize=(25, 25),)
            for (xx, yy, ww, hh) in smile:
                cv2.rectangle(b, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

            cv2.imshow('webcam', img)
            if cv2.waitKey(1) and 0xff ==ord("q"):
                break
    except Exception as er:
        print(''.format(er))



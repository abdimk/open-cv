from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import json

faceCascade = cv2.CascadeClassifier('modelname.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 200)
current_time = datetime.now().time()
current_date = datetime.now().date()
while True:
    try:
        timer = cv2.getTickCount()
        value, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(20, 20))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)


        for (x, y, w, h) in faces:
            cv2.putText(img,"modelname",(x - 60,y - 60),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
            cv2.putText(img,str(fps),(x - 20, y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(225,0,0),2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
        path = Path("modelname")
        def vison(control):
            with open(f"{path.stem}", "w") as file:
                file.write(f"{path.stem}, {current_time}, {current_date}, visible = {control}")
        vison(value)
        cv2.imshow('video', img)
        if cv2.waitKey(1) and 0xff==ord("q"):
            break

    except Exception as er:
        print(' '.format(er))
cap.release()
cv2.destroyAllWindows()

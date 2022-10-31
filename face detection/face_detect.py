# -*- coding: utf-8 -*-
"""
Created on Nov 2019

@author: mohamed mohamed taha
"""



#import pyautogui
#import time
##x=1



#import numpy as np
import cv2
video = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#cap = cv2.VideoCapture(0)

while True:

    ret,img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #img =pyautogui.screenshot('D:/Oliver/python/projects/image1' + '.png')

    cv2.imshow('frame',img)
#    cv2.imshow=pyautogui.screenshot('D:/Oliver/python/projects/image1' + '.png')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()

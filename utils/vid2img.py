import cv2
import os

vid = cv2.VideoCapture("IMG_0537.MOV")
index = 0
ret = 1

while ret:
    ret, frame = vid.read()
    
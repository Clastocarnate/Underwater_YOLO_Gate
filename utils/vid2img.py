import cv2
import os

vid = cv2.VideoCapture("IMG_0537.MOV")
index = 1
ret = 1

while ret:
    ret, frame = vid.read()
    cv2.imwrite(f"dataset/{index}.jpg",frame)
    index += 1
    
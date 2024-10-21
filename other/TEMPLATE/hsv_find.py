# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
        print("HSVmin :  [%d,%d,%d]" % (HSV[y,x][0] - error,HSV[y,x][1] - error,HSV[y,x][2] - error))
        print("HSVmax :  [%d,%d,%d]\n" % (HSV[y,x][0] + error,HSV[y,x][1] + error,HSV[y,x][2] + error))


if __name__ == "__main__":
    print("请输入 范围差值 :\n")
    error = int(input())

    temp_dir = "resource/template"
    vedio_dir = "3.mp4"
    capture = cv2.VideoCapture(vedio_dir)
    size_percent = 100
    start_frame = 25000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    threshold = 50  # Initial threshold value
    count = 0
    while True:
        ret, image = capture.read()
        HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        cv2.imshow("imageHSV",HSV)
        cv2.imshow('image',image)
        cv2.setMouseCallback("imageHSV",getpos)
        cv2.waitKey(0)
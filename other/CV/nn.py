import numpy as np
import cv2
from matplotlib import pyplot as plt


def ORB(img):
    """
     ORB角点检测
     实例化ORB对象
    """
    orb = cv2.ORB_create(nfeatures=500)
    """检测关键点，计算特征描述符"""
    kp, des = orb.detectAndCompute(img, None)

    # 将关键点绘制在图像上
    img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    cv2.imwrite("1.jpg", img2)

    # 画图
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(img2[:, :, ::-1])
    plt.xticks([]), plt.yticks([])
    plt.show()
    return kp, des


img1 = cv2.imread("00000.jpg")
kp1, des1 = ORB(img1)


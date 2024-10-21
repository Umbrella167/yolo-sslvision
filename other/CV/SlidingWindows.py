import cv2
import numpy as np
import time
from scipy.spatial import cKDTree

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def on_trackbar(val):
    global threshold
    threshold = val / 100


def sliding_window(image, step_size, window_size):
    """
    滑动窗口函数

    :param image: 输入图像
    :param step_size: 窗口移动的步长
    :param window_size: 窗口的大小 (宽, 高)
    :return: 返回每个窗口的 (x, y, 窗口) 的生成器
    """
    # 遍历图像的每个窗口
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            # 提取窗口
            window = image[y:y + window_size[1], x:x + window_size[0]]
            yield (x, y, window)

def filter_points(data, r, N):
    '''
    r 这是一个半径 用于检查每个数据点周围的邻居点。
    N 这是一个阈值 表示在半径 r 内需要有的点的最小数量，以便将该点包括在输出中。  
    '''
    tree = cKDTree(data)
    filtered_points = []
    centers = []
    bounding_boxes = []

    for point in data:
        indices = tree.query_ball_point(point, r)
        if len(indices) >= N:
            group_points = data[indices]
            filtered_points.append(point)

            # Calculate the center of the group
            center = np.mean(group_points, axis=0)
            centers.append(center)

            # Calculate the bounding box (min and max for each dimension)
            min_point = np.min(group_points, axis=0)
            max_point = np.max(group_points, axis=0)
            bounding_box = (min_point, max_point)
            bounding_boxes.append(bounding_box)

    return np.array(filtered_points), np.array(centers), bounding_boxes

def equalize_color_image(image):
    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 对V通道进行直方图均衡化
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    # 将图像从HSV转换回BGR
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_img

if __name__ == "__main__":
    temp_dir = "resource/template"
    vedio_dir = "3.mp4"
    capture = cv2.VideoCapture(vedio_dir)
    
    size_percent = 100
    start_frame = 25000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    threshold = 50  # Initial threshold value
    lower_color = np.array([77,13,10], dtype=np.uint8)
    upper_color = np.array([117,53,255], dtype=np.uint8)
    count = 0
    image_size = (1080, 720)
    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)

        frame = cv2.GaussianBlur(frame, ksize=(5,5), sigmaX=1)
        # 增加彩色对比度
        frame = equalize_color_image(frame)
        # edges = cv2.Canny(frame, 20, 100)
        edges = cv2.Canny(frame, 220, 255)

        window = frame.copy()
        width, height = image_size
        # window_size = (int(width * 0.5), int(height * 0.5))
        window_size = (10, 10)

        # corners = cv2.goodFeaturesToTrack(edges, 0, 0.5, 1, mask=None, blockSize=3, useHarrisDetector=False, k=0.04)
        # if corners is None:
        #     continue
        # corners = corners.reshape(-1, 2).astype(int)  
        # corners,centers,bounding_boxes = filter_points(corners,50,3)
        # for i in bounding_boxes:
        #     cx, cy = i[0].ravel()
        #     w,h = i[1].ravel()
        #     cv2.rectangle(frame, (cx, cy), (w, h), (0, 255, 0), 2)

        #     # cv2.circle(window, (int(cx), int(cy)), 3, (255, 0, 0), -1)


        for (x, y, window) in sliding_window(edges, 100, window_size):
            # corners = cv2.goodFeaturesToTrack(window, 0, 0.5, 1, mask=None, blockSize=3, useHarrisDetector=False, k=0.04)
            # if corners is not None:
            #     corners = corners.astype(int)  
            #     for i in corners:  
            #         cx, cy = i.ravel()
            #         if len(window.shape) == 2 or window.shape[2] == 1:
            #             window = cv2.cvtColor(window, cv2.COLOR_GRAY2BGR)
            #         cv2.circle(window, (cx, cy), 3, (255, 0, 0), -1)
            cv2.imshow("res",window)
        if cv2.waitKey(0) & 0xFF == ord('q'): 
            break
        end_tiem = time.time()
        print(f"FPS:{1/(end_tiem-start_time):.2f}")

    capture.release()
    cv2.destroyAllWindows()


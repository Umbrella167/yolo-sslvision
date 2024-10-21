import dearpygui.dearpygui as dpg
import math
import numpy as np
import os
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor

def save_frame(frame, file_name, quality):
    if not cv2.imwrite(file_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        print(f"Error: Failed to write image {file_name}.")

def generate_images(path, start_frame, end_frame, step, quality, size=None, output_dir="resource/output_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Unable to read video at the specified path.")
        return

    # Set the starting position of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    saved_frame_count = 0  # Track how many frames have been saved

    with ThreadPoolExecutor(max_workers=4) as executor:  # 使用线程池来并行保存图像
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Frame {current_frame} could not be read. Stopping extraction.")
                break

            if (current_frame - start_frame) % step == 0:
                if size:
                    frame = cv2.resize(frame, size)
                file_name = os.path.join(output_dir, f"{saved_frame_count:05d}.jpg")
                executor.submit(save_frame, frame, file_name, quality)
                saved_frame_count += 1

            current_frame += 1

    cap.release()
    print("Image extraction completed.")

def format_time(time):
    if time < 60:
        return f"{time}秒"
    elif time < 3600:
        minutes = time // 60
        seconds = time % 60
        return f"{minutes}分 {seconds}秒"
    else:
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = time % 60
        return f"{hours}时 {minutes}分 {seconds}秒"

def calculate_distance( pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
def middle_pos( pos1, pos2):
    return [(pos1[0] + pos2[0]) / 2,(pos1[1] + pos2[1]) / 2]
def calculate_center_point(points):
    """
    计算四边形的中心点
    :param points: 四边形四个点的坐标列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :return: 四边形中心点的坐标 (x, y)
    """
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    center_x = sum(x_coords) / 4
    center_y = sum(y_coords) / 4
    
    return center_x, center_y

def apply_transform(matrix, point):
    # 将 DearPyGui 矩阵转换为 NumPy 矩阵
    np_matrix = np.array(matrix).reshape(3, 3)
    # 确保 point 是 [x, y, 1]
    point = np.array([point[0], point[1], 1])
    # 进行矩阵乘法
    transformed_point = np_matrix @ point
    return transformed_point[:2]  # 返回 [x, y]

def matrix2list_mouse(matrix):
    transform = []
    for i in range(16):
        transform.append(matrix[i])
    data_array = np.array(transform)
    matrix = data_array.reshape(4, 4)
    matrix[0, 3] = -1 * matrix[-1, 0]
    matrix[1, 3] = -1 * matrix[-1, 1]
    matrix[-1, 0] = 0
    matrix[-1, 1] = 0
    return np.array(matrix)
def matrix2list(matrix):
    transform = []
    for i in range(16):
        transform.append(matrix[i])
    data_array = np.array(transform)
    matrix = data_array.reshape(4, 4)
    matrix[0, 3] = matrix[-1, 0]
    matrix[1, 3] = matrix[-1, 1]
    matrix[-1, 0] = 0
    matrix[-1, 1] = 0
    return np.array(matrix)

def mouse2ssl(x,y,translation_matrix,scale):
    x1,y1 = (matrix2list_mouse(translation_matrix) @ np.array([x,y,1,1]))[:2]
    return int(x1 / scale),int(-1 * y1 / scale)


def swap_elements(lst, element1, element2):
    try:
        # 找到元素的索引
        index1 = lst.index(element1)
        index2 = lst.index(element2)
        # 交换元素
        lst[index1], lst[index2] = lst[index2], lst[index1]
    except ValueError:
        print("其中一个元素不在列表中")
def compare_dicts(dict1, dict2):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    added_keys = keys2 - keys1
    removed_keys = keys1 - keys2
    common_keys = keys1 & keys2
    modified_items = {key: dict2[key] for key in common_keys if dict1[key] != dict2[key]}
    return added_keys, removed_keys, modified_items

def image2texture(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    texture_data = image.ravel().astype('float32') / 255
    return texture_data
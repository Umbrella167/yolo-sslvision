import cv2
import numpy as np
import math
from SAM2 import sam2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
class Utils:
    def __init__(self):
        self.color_list = {
            "yellow": (0, 255, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "pink": (255, 0, 255),
        }
        self.color_id = {
            "yellow": 0,
            "blue": 1,
            "green": 2,
            "pink": 3,
        }
        pass
    def map_coordinates_to_original(self,point,rect_data , scale_x = 35,scale_y = 35):
        # Calculate the scale factors
        top_left_x, top_left_y, w, h = rect_data
        x_robot, y_robot = point
        scale_x = w / scale_x
        scale_y = h / scale_y
        
        # Map the coordinates back
        x_original_in_crop = x_robot * scale_x
        y_original_in_crop = y_robot * scale_y
        
        # Add the top-left corner offset
        x_original = top_left_x + x_original_in_crop
        y_original = top_left_y + y_original_in_crop
        
        return int(x_original), int(y_original)

    # 获取每一类坐标最接近的颜色
    def assign_color(self, color_class: dict, robot_data: list,rect_data:list):
        vision_data = {}

        def get_center_color(color):
            b, g, r = color
            if (b < r) and (b < g):
                return "yellow"
            else:
                return "blue"

        def get_side_color(color):
            b, g, r = color
            if (g > r) and (g > b):
                return "green"
            else:
                return "pink"

        for index in color_class:
            _len = len(color_class[index])
            center = self.find_most_central_point(robot_data)
            __robot_data = []
            for color_data in color_class[index]:

                point = color_data[0][0]
                radius = color_data[0][1]
                color = color_data[-1]
                if point == center:
                    closest = get_center_color(color)
                else:
                    closest = get_side_color(color)
                origin_point = utils.map_coordinates_to_original(point,rect_data)
                __robot_data.append({"origin_point":origin_point,"point":point, "radius":radius, "color":color})

            vision_data[closest] = __robot_data
        return vision_data

    # 匹配字典中最接近的颜色
    def find_closest_template_color(self, template_color, input_color):
        def color_distance(c1, c2):
            # Calculate the Euclidean distance between two colors
            return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

        closest_color = None
        min_distance = float("inf")
        for color_name, color_value in template_color.items():
            distance = color_distance(input_color[::-1], color_value)
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        return closest_color

    # 将point对应的颜色分成k类
    def cluster_image_colors(self, image, points, n):
        # 提取坐标点对应的颜色
        colors = []
        for point in points:
            x, y = point[0]
            # 获取图像中指定坐标的颜色值
            color = image[y, x]
            colors.append(color)
        # 将颜色转换为numpy数组
        colors = np.array(colors)
        # 使用KMeans聚类算法将颜色分为n类
        kmeans = KMeans(n_clusters=n, random_state=0).fit(colors)
        labels = kmeans.labels_
        # 将坐标点按颜色类别分组
        res = {}
        for i in range(n):
            res[f"{i+1}"] = res[f"{i+1}"] = [
                [points[j], image[points[j][0][::-1]]]
                for j in range(len(points))
                if labels[j] == i
            ]
        return res

    # 寻找最佳k
    def find_optimal_clusters(self, image, points, max_k=3):
        # 提取坐标点对应的颜色
        colors = []
        for point in points:
            x, y = point[0]
            color = image[y, x]
            colors.append(color)

        colors = np.array(colors)

        sse = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(colors)
            sse.append(kmeans.inertia_)

        sse_diff = np.diff(sse)
        optimal_k = len(sse)
        if abs(sse_diff[-1]) < 300:
            optimal_k = optimal_k - 1
        return optimal_k

    # 判断是否是唯一圆
    def is_circle_unique(self, circles: list, circle):
        """
        判断给定的圆是否在已有的圆列表中是唯一的。

        参数:
        - circles: 已有圆的列表，其中每个圆由中心坐标和半径组成。
        - circle: 要检查的圆，由中心坐标和半径组成。

        返回:
        - 如果该圆与列表中的任何圆不相似，返回 True；否则返回 False。
        """
        for existing_circle in circles:
            center, radius = existing_circle
            if utils.calculate_distance(center, circle[0]) < radius:
                return False
        return True

    def read_rectangles_from_file(self, filename, height, width):
        rectangles = []
        with open(filename, "r") as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    parts = line.split()
                    if len(parts) == 5:
                        # Extract the rectangle parameters
                        _, x_center, y_center, width_ratio, height_ratio = map(
                            float, parts
                        )
                        rectangles.append(
                            (
                                x_center * width,
                                y_center * height,
                                width_ratio * width,
                                height_ratio * height,
                            )
                        )
        return rectangles

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # 计算一个点到其他所有点的平均距离
    def average_distance_to_other_points(self, point, points):
        total_distance = 0
        for other_point in points:
            total_distance += self.calculate_distance(point, other_point)
        return total_distance / len(points)

    # 寻找五个点中最中间的点
    def find_most_central_point(self, points_with_radius):
        # 提取点列表
        points = [point_with_radius[0] for point_with_radius in points_with_radius]

        # 初始化最小平均距离和最中间的点
        min_average_distance = float("inf")
        most_central_point = None

        # 找到平均距离最小的点
        for point in points:
            avg_distance = self.average_distance_to_other_points(point, points)
            if avg_distance < min_average_distance:
                min_average_distance = avg_distance
                most_central_point = point
        return most_central_point

    # 计算四边形的中点
    def calculate_centroid_of_quadrilateral(self, points):
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return [centroid_x, centroid_y]

utils = Utils()

class HoughCircles:
    def __init__(self):
        pass

    def get_circles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.8)
        gray = cv2.resize(gray, (500, 500))
        canny = cv2.Canny(gray, 0, 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            2,
            150,
            param1=10,
            param2=50,
            minRadius=10,
            maxRadius=80,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(gray, (i[0], i[1]), i[2], (255, 255, 255), 2)
                # Draw the center of the circle
                cv2.circle(gray, (i[0], i[1]), 2, (255, 255, 255), 3)
        return gray


class RandomRules:
    def __init__(self):
        self.robot_list = []

    def magic_mask_sam2(self, image, point):
        masks, scores, logits = sam2.add_point([point], [1])
        return masks[0]

    def magic_mask(self, image, point, color_buffer, dist):
        # 确保输入点在图像范围内
        if (
            point[0] < 0
            or point[0] >= image.shape[1]
            or point[1] < 0
            or point[1] >= image.shape[0]
        ):
            raise ValueError("Point is outside the image boundaries.")

        # 将图像从 RGB 转换为 HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 获取目标颜色
        target_color = hsv_image[point[1], point[0], :]

        # 设置颜色缓冲区
        color_buffer = [color_buffer, color_buffer, color_buffer]

        # 计算颜色范围
        lower_bound = np.clip(target_color - color_buffer, 0, 255)
        upper_bound = np.clip(target_color + color_buffer, 0, 255)
        # 创建颜色掩码
        color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 计算每个像素到起始点的距离
        h, w = image.shape[:2]
        y_indices, x_indices = np.indices((h, w))
        distances = np.sqrt((x_indices - point[0]) ** 2 + (y_indices - point[1]) ** 2)

        # 创建距离掩码
        distance_mask = distances <= dist

        # 结合颜色掩码和距离掩码
        final_mask = cv2.bitwise_and(
            color_mask, color_mask, mask=distance_mask.astype(np.uint8)
        )

        # 查找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8
        )

        # 检查是否有连通区域
        if num_labels <= 1:
            return np.zeros_like(final_mask)  # 返回全零掩码

        # 找到面积最大的连通区域
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景

        # 创建最终掩码,仅保留最大区域
        largest_mask = np.zeros_like(final_mask)
        largest_mask[labels == largest_label] = 255

        return largest_mask

    def is_circle(self, mask, image, circularity_threshold=0.8, fill=True):
        """
        检测蒙版图像中最大的连通区域是否为圆形,并在原始图像上绘制检测到的圆形轮廓

        参数:
        - mask: 二值化的蒙版图像,用于轮廓检测
        - image: 原始图像,用于在检测到圆形后绘制轮廓
        - circularity_threshold: 圆形度的阈值,默认为0.8
        - fill: 布尔值,指示是否填充圆形轮廓,默认为True
        返回:
        - 圆形的概率值（0到1之间）
        - 修改后的图像
        - 轮廓的中心坐标 (x, y) 和半径
        """
        confidence = 0.0  # 初始化圆形的概率值
        result_image = image.copy()  # 初始化返回的图像
        center = (0, 0)  # 初始化中心坐标
        radius = 0  # 初始化半径

        if mask.dtype != np.uint8:
            mask = cv2.convertScaleAbs(mask)

        # 找到蒙版中的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果找到了轮廓
        if contours:
            # 计算图像总面积
            image_area = image.shape[0] * image.shape[1]

            # 过滤掉面积大于 50% 和小于 10% 的轮廓
            valid_contours = [
                cnt
                for cnt in contours
                if 0.01 * image_area <= cv2.contourArea(cnt) <= 0.3 * image_area
            ]

            # 如果找到了符合条件的轮廓
            if valid_contours:
                # 假设最大的轮廓是最大的连通区域
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # 计算最大轮廓的面积
                area = cv2.contourArea(largest_contour)

                # 计算最大轮廓的周长
                perimeter = cv2.arcLength(largest_contour, True)

                # 避免除以零
                if perimeter != 0:
                    # 计算圆形度
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))

                    # 计算圆形的概率
                    confidence = min(circularity / circularity_threshold, 1.0)

                    # 计算最小包围圆的中心和半径
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    center = (int(x), int(y))
                    radius = int(radius)

                    # 如果是圆形,绘制轮廓
                    if circularity >= circularity_threshold:
                        thickness = (
                            -1 if fill else 1
                        )  # 如果填充设为True,线条宽度设为-1,否则为1
                        cv2.drawContours(
                            result_image, [largest_contour], -1, (0, 255, 0), thickness
                        )  # 绿色

        return confidence, result_image, center, radius

    def get_robot_circle_sam2(self, image):
        circle_list = []
        sam2.load_image(image)
        height, width, _ = image.shape
        step = 3
        for y in range(0, height, step):
            for x in range(0, width, step):
                point = [x, y]
                mask = self.magic_mask_sam2(image, point)
                confidence, circle, center, radius = self.is_circle(mask, image)
                if confidence > 0.99:
                    circle_data = [center] + [radius]
                    if utils.is_circle_unique(circle_list, circle_data):
                        circle_list.append(circle_data)

                if len(circle_list) >= 5:

                    _center = utils.find_most_central_point(circle_list)
                    side_points = [
                        point[0] for point in circle_list if point[0] != _center
                    ]
                    __centet_x, __centet_y = utils.calculate_centroid_of_quadrilateral(
                        side_points
                    )
                    dist = utils.calculate_distance(_center, (__centet_x, __centet_y))
                    if dist > 4:
                        return circle_list
                    for circle in circle_list:
                        image = cv2.circle(image, circle[0], circle[1], (0, 255, 0), -1)
                    self.robot_list.append(circle_list)
                    return circle_list
        return circle_list

    def get_mark_data(self, image, rectangles,robot_quantity = 9999):
        height, width, _ = image.shape
        mark_data = []

        for count, (x, y, w, h) in enumerate(rectangles):
            if count >= robot_quantity:
                break
            top_left_x, top_left_y = (max(0, x - w // 2), max(0, y - h // 2))
            bottom_right_x, bottom_right_y = (
                min(width, top_left_x + w),
                min(height, top_left_y + h),
            )
            image_robot = image[
                int(top_left_y) : int(bottom_right_y),
                int(top_left_x) : int(bottom_right_x),
            ]
            resize_robot = (35,35)
            image_robot = cv2.resize(image_robot, resize_robot)
            image_robot_yuv = cv2.cvtColor(image_robot, cv2.COLOR_BGR2YUV)
            robot_data = self.get_robot_circle_sam2(image_robot_yuv)
            if len(robot_data) >= 3:
                k = utils.find_optimal_clusters(image_robot, robot_data, 3)
                res = utils.cluster_image_colors(image_robot, robot_data, k)
                __robot_data = utils.assign_color(
                    res,
                    robot_data,(top_left_x, top_left_y,w,h)
                )
                print(f"Robot:{count} Success! ")
                mark_data.append(__robot_data)
        return mark_data
    
if __name__ == "__main__":
    randomrules = RandomRules()
    # 定义输入数据集路径
    image_base_path = "datasets/images/"
    label_base_path = "datasets/labels/"

    # 定义输出数据集路径
    output_image_base_path = "resource/datasets/images/"
    output_label_base_path = "resource/datasets/labels/"
    
    # 确保输出目录存在
    os.makedirs(output_image_base_path, exist_ok=True)
    os.makedirs(output_label_base_path, exist_ok=True)

    # 遍历数据集的 train 和 val 目录
    for dataset_type in ["train", "val"]:
        image_dir = os.path.join(image_base_path, dataset_type)
        label_dir = os.path.join(label_base_path, dataset_type)

        output_image_dir = os.path.join(output_image_base_path, dataset_type)
        output_label_dir = os.path.join(output_label_base_path, dataset_type)

        # 确保输出子目录存在
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # 获取该目录下所有图像文件名
        image_files = os.listdir(image_dir)

        for image_file in image_files:
            # 构造完整的图像路径和对应的标签文件路径
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

            # 读取图像
            image = cv2.imread(image_path)
            frame = image.copy()
            
            # 获取图像的尺寸
            height, width, _ = image.shape

            # 读取矩形数据
            rectangles = utils.read_rectangles_from_file(label_path, height, width)

            # 处理数据
            print("Image:",image_file,"   Start!")
            mark_data = randomrules.get_mark_data(image, rectangles)

            # 保存处理后的图像和数据
            output_image_path = os.path.join(output_image_dir, image_file)
            output_label_path = os.path.join(output_label_dir, image_file.replace(".jpg", ".txt"))

            # 这里假设我们仅将处理后的图像保存
            cv2.imwrite(output_image_path, image)
            # 保存处理后的矩形数据并进行归一化
            with open(output_label_path, 'w') as f:
                for robot_data in mark_data:
                    for stand_color in robot_data:
                        for once_color in robot_data[stand_color]:
                            point = once_color["origin_point"]
                            radius = once_color["radius"]
                            obj_id = utils.color_id[stand_color]
                            x, y = point
                            w, h = (radius, radius)
                            
                            frame = cv2.circle(frame, (x, y), radius, utils.color_list[stand_color], -1)                                        
                            # 归一化                      
                            x_norm = x / width
                            y_norm = y / height
                            w_norm = w / width               
                            h_norm = h / height
                            
                            # 写入归一化的数据到文件
                            f.write(f"{obj_id} {x_norm} {y_norm} {w_norm} {h_norm}\n")
    
            # 显示结果图像
            cv2.imshow("res", frame)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
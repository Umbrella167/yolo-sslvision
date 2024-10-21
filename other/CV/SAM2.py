import os
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
class SAM2Image:
    def __init__(self):
        # self.batch_size = batch_size
        self.color = [
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 红色
            (0, 0, 255),      # 蓝色
            (255, 255, 0),    # 黄色
            (255, 0, 255),    # 洋红
            (0, 255, 255),    # 青色
            (255, 255, 255),  # 白色
            (0, 0, 0),        # 黑色
            (128, 0, 0),      # 栗色
            (128, 128, 0),    # 橄榄色
            (0, 128, 0),      # 深绿色
            (128, 0, 128),    # 紫色
            (0, 128, 128),    # 蓝绿色
            (0, 0, 128),      # 海军蓝
            (192, 192, 192),  # 银色
            (128, 128, 128),  # 灰色
            (255, 165, 0),    # 橙色
            (255, 192, 203),  # 粉红色
            (210, 105, 30),   # 巧克力色
            (75, 0, 130),     # 靛蓝
            (173, 216, 230),  # 淡蓝色
            (245, 222, 179),  # 小麦色
            (139, 69, 19),    # 棕色
            (255, 20, 147),   # 深粉红
            (64, 224, 208),   # 绿松石
            (0, 255, 127),    # 春绿色
            (255, 69, 0),     # 橙红色
            (154, 205, 50),   # 黄绿色
            (238, 130, 238),  # 紫罗兰
            (127, 255, 212),  # 绿宝石
            (100, 149, 237),  # 矢车菊蓝
            (255, 215, 0),    # 金色
            (220, 20, 60),    # 猩红
            (46, 139, 87),    # 海洋绿
            (0, 100, 0),      # 深绿
            (72, 61, 139),    # 暗紫
            (255, 99, 71),    # 番茄
            (32, 178, 170),   # 浅海洋绿
            (135, 206, 235),  # 天空蓝
            (250, 128, 114),  # 鲑鱼色
            (219, 112, 147),  # 苍紫罗兰红
            (244, 164, 96)    # 沙棕色
        ]
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.init_moudel()

    def init_moudel(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(1).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        self.sam2_checkpoint = "resource/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def load_image(self,image):
        self.predictor.set_image(image)
        
    def add_point(self,points,labels):
        points = np.array(points)
        labels = np.array(labels)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return masks, scores, logits

    def add_positive_point(self,points):
        points = np.array(points)
        labels = np.ones(len(points))
        return self.add_point(points,labels)
    def add_negative_point(self,points):
        points = np.array(points)
        labels = np.zeros(len(points))
        return self.add_point(points,labels)
    def mask2rect(self, masks, frame, min_area=0):
        # Find contours
        rects = []
        image = frame.copy()
        for index, mask in enumerate(masks):
            # Convert mask to uint8
            mask_uint8 = mask.astype(np.uint8) * 255  # Scale to [0, 255]
            color = self.color[index]
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x_center, y_center, w_norm, h_norm = 0, 0, 0, 0
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                height, width, _ = image.shape
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                rect = [x_center, y_center, w_norm, h_norm]
                rects.append(rect)
        return rects, image

sam2 = SAM2Image()
if __name__ == "__main__":
    sam2 = SAM2Image()
    image = cv2.imread("00000.jpg")
    sam2.load_image(image)
    points = [[0, 0]]
    labels = [1]
    masks, scores, logits = sam2.add_point(points, labels)
    rect,mask = sam2.mask2rect(masks, image)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
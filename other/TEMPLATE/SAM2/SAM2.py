import os
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import cv2
import random
import yaml

class SAM2:

    def __init__(self, video_dir = "resource/output_images"):
        # self.batch_size = batch_size
        self.las_mask_len = 0
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
        self.video_dir = video_dir
        self.predict_path = f"{self.video_dir}/predict"
        self.init_moudel()
        try:
            self.inference_state = self.predictor.init_state(video_path=self.video_dir,async_loading_frames=True)
            # self.predictor.reset_state(self.inference_state)
        except:
            print("Video not found")

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
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, device=self.device
        )
    
    def add_positive_point(self, points, frame_id, obj_id):
        labels = np.array([1 for i in range(len(points))], np.int32)
        
        _, out_obj_ids, out_mask_logits = self.add_point(points, frame_id, obj_id, labels)
        return out_obj_ids, out_mask_logits

    def add_negative_point(self, points, frame_id, obj_id):
        labels = np.array([0 for i in range(len(points))], np.int32)
        _, out_obj_ids, out_mask_logits = self.add_point(points, frame_id, obj_id, labels)
        return out_obj_ids, out_mask_logits
    
    def add_point(self,points,frame_id,obj_id,labels):
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_id,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_obj_ids, out_mask_logits

    def _logits2mask(self, logits, obj_id = [0],res=255, threshold=0.0):
        binary_mask = []
        for i in range(len(obj_id)):
            out_mask_logits_np = logits[i].cpu().numpy()
            binary_mask.append(((out_mask_logits_np > threshold).astype(np.uint8) * res)[0])
        return binary_mask
    def overlay_mask_on_image(self,image, masks, alpha=0.5, color=(0, 255, 0)):
        """
        Overlay masks on the original image.

        :param image: Original image (numpy array)
        :param masks: List of binary masks (numpy arrays)
        :param alpha: Transparency factor for the overlay
        :param color: Color to use for the mask overlay (B, G, R)
        :return: Image with mask overlay
        """
        # Convert image to RGB if it is in grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create a copy of the original image to overlay the masks
        overlay = image.copy()

        for mask in masks:
            # Ensure mask is binary
            mask = mask.astype(bool)

            # Apply the color to the mask
            colored_mask = np.zeros_like(overlay, dtype=np.uint8)
            colored_mask[mask] = color

            # Add the colored mask to the overlay
            overlay = cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)

        return overlay
    def generate_all(self):
        torch.cuda.empty_cache()
        video_segments = {}
        for (out_frame_idx, out_obj_ids, out_mask_logits) in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def mask2rect(self, masks, frame, min_area=0, is_save=True, template_folder='resource/template'):
        # Ensure the template folder exists
        maks_len = len(masks)
        os.makedirs(template_folder, exist_ok=True)
        
        # Find the current maximum index from existing images in the folder
        existing_files = os.listdir(template_folder)
        image_indices = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg')]
        current_index = max(image_indices) + 1 if image_indices else 0

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
                
                # Save the rectangle image if is_save is True
        if is_save :
            rect_image = frame[y:y+h, x:x+w]
            rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2RGB)
            if maks_len == self.las_mask_len:
                current_index -= 1
            save_path = os.path.join(template_folder, f"{current_index}.jpg")
            cv2.imwrite(save_path, rect_image)
            current_index += 1
        self.las_mask_len = maks_len
        return rects, image


    def generate_yolo_datas(self, is_continue: bool = True, rate_train: float = 0.8):
        max_frame_id = 0
        
        output_dir = os.path.join("resource/datasets")
        if not is_continue:
            os.path.exists(output_dir) and os.system("rm -rf datasets")
        else:
            # Read the existing max frame_id from train and val directories
            for set_type in ['train', 'val']:
                images_dir = os.path.join(output_dir, 'images', set_type)
                if os.path.exists(images_dir):
                    existing_files = os.listdir(images_dir)
                    # Extract frame ids from filenames and find the maximum
                    for file in existing_files:
                        if file.endswith('.jpg'):
                            frame_id = int(file.split('.')[0])
                            max_frame_id = max(max_frame_id, frame_id)
        
        video_segments = self.generate_all()

        # Define directory structure
        subdirs = {"images": ["train", "val"], "labels": ["train", "val"]}

        # Create directories
        for subdir, types in subdirs.items():
            for t in types:
                dir_path = os.path.join(output_dir, subdir, t)
                os.makedirs(dir_path, exist_ok=True)

        # Generate data
        for frame_id in video_segments:
            frame_path = os.path.join(self.video_dir, f"{str(frame_id).zfill(5)}.jpg")
            frame = cv2.imread(frame_path)
            
            # Collect labels for all objects in the current frame
            labels = []
            for obj_id in video_segments[frame_id]:
                mask = video_segments[frame_id][obj_id]
                rects, _ = self.mask2rect(mask, frame, 30)
                for rect in rects:
                    labels.append([obj_id] + rect)
            
            # Decide whether this image goes to the train or val directory
            set_type = "train" if random.random() < rate_train else "val"

            # Adjust frame_id by adding the max_frame_id if continuing
            new_frame_id = frame_id + max_frame_id

            # Define paths for saving image and labels
            image_name = f"{str(new_frame_id).zfill(5)}.jpg"
            label_name = f"{str(new_frame_id).zfill(5)}.txt"
            image_output_path = os.path.join(output_dir, "images", set_type, image_name)
            label_output_path = os.path.join(output_dir, "labels", set_type, label_name)

            # Save the image
            cv2.imwrite(image_output_path, frame)
            # Save the labels
            with open(label_output_path, "w") as label_file:
                for label in labels:
                    label_file.write(" ".join(map(str, label)) + "\n")

    def generate_yaml_file(
        self, num_classes, class_names, yaml_path="resource/yolo.yaml"
    ):
        """
        Generates a YAML configuration file for YOLOv8 training.

        :param num_classes: Number of classes in the dataset.
        :param class_names: List of class names.
        :param yaml_path: Path where the YAML file will be saved.
        """
        data_yaml = {
            "train": "images/train",
            "val": "images/val",
            "nc": num_classes,
            "names": class_names,
        }

        with open(yaml_path, "w") as file:
            yaml.dump(data_yaml, file, default_flow_style=False)
    
    def get_max_index(self):
        path = "temp_frames"
        files = os.listdir(path)
        files.sort()
        return int(files[-1].split(".")[0])

    def reset(self):
        self.predictor.reset_state(self.inference_state)

sam2 = SAM2()
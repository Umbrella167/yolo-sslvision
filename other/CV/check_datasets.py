import os
import cv2
import shutil

def draw_labels_on_images(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt'))
        
        if not os.path.isfile(label_path):
            continue
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        height, width, _ = image.shape
        
        with open(label_path, 'r') as label_file:
            for line in label_file:
                parts = line.strip().split()
                # Assuming the label format is: class_id x_center y_center width height (normalized)
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                
                # Convert from normalized to image coordinates
                left = int((x_center - box_width / 2) * width)
                top = int((y_center - box_height / 2) * height)
                right = int((x_center + box_width / 2) * width)
                bottom = int((y_center + box_height / 2) * height)
                
                # Draw rectangle on the image
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, image)
#生成新的数据集
def generate_augmented_dataset(image_folder, label_folder, output_folder='CNN_DATA'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    index = 0
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt'))
        
        if not os.path.isfile(label_path):
            continue
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        height, width, _ = image.shape
        
        with open(label_path, 'r') as label_file:
            for line in label_file:
                parts = line.strip().split()
                # Assuming the label format is: class_id x_center y_center width height (normalized)
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
                
                # Convert from normalized to image coordinates
                left = int((x_center - box_width / 2) * width)
                top = int((y_center - box_height / 2) * height)
                right = int((x_center + box_width / 2) * width)
                bottom = int((y_center + box_height / 2) * height)
                
                # Extract the rectangle from the image
                cropped_image = image[top:bottom, left:right]

                # Save the cropped image with the format class_id_index.jpg
                output_filename = f"{str(index).zfill(5)}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, cropped_image)
                index += 1  # Increment index for the next image
def update_dataset_after_deletion(image_folder, label_folder, visualized_folder):
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        label_path = os.path.join(label_folder, image_filename.replace('.jpg', '.txt'))
        visualized_path = os.path.join(visualized_folder, image_filename)
        
        if not os.path.exists(visualized_path):
            # If the visualized image is deleted, remove the corresponding image and label
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)

# update_dataset_after_deletion('resource/datasets/images/val', 'resource/datasets/labels/val', 'resource/datasets/visualized/val')
# update_dataset_after_deletion('resource/datasets/images/train', 'resource/datasets/labels/train', 'resource/datasets/visualized/train')


# draw_labels_on_images('resource/datasets/images/val', 'resource/datasets/labels/val', 'resource/datasets/visualized/val')
# draw_labels_on_images('resource/datasets/images/train', 'resource/datasets/labels/train', 'resource/datasets/visualized/train')

generate_augmented_dataset('resource/datasets/images/train', 'resource/datasets/labels/train', 'CNN_DATA/train')
generate_augmented_dataset('resource/datasets/images/val', 'resource/datasets/labels/val', 'CNN_DATA/val')

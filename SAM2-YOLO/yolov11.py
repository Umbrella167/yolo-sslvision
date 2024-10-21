from ultralytics import YOLO
import cv2
import time

def train():
    model = YOLO("resource/yolo11n.pt")  # 加载模型
    trainer = model.train(data="resource/data.yaml", epochs=200)  # 训练模型

# 调用训练函数

if __name__ == "__main__":
    color_list = [(255,0,0),(0,255,255),(0,255,0),(255,0,255)]
    # Load the trained models
    model_robot = YOLO("runs/detect/train6/weights/best.pt")  # Ensure the path is correct
    model_color = YOLO("runs_color/detect/train4/weights/best.pt")  # Ensure the path is correct

    # Open the video file
    capture = cv2.VideoCapture('resource/3.mp4')
    start_frame = 45000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1080, 720))
        # Perform prediction on the current frame using model_robot
        results = model_robot.predict(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Coordinates of the bounding box
                confidence = box.conf  # Confidence score
                class_id = box.cls  # Class ID
                # Crop the rectangle (bounding box) from the frame
                cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                # Perform prediction on the cropped image using model_color
                color_results = model_color.predict(cropped_image)
                # Process color_results as needed
                for color_result in color_results:
                    for color_box in color_result.boxes:
                        # Extract color prediction details
                        cx1, cy1, cx2, cy2 = color_box.xyxy.tolist()[0]
                        color_confidence = float(color_box.conf)
                        color_class_id = int(color_box.cls)
                        # Translate cropped box coordinates to original frame coordinates
                        original_cx1 = int(cx1 + x1)
                        original_cy1 = int(cy1 + y1)
                        original_cx2 = int(cx2 + x1)
                        original_cy2 = int(cy2 + y1)
                        # Draw the color prediction on the original frame
                        cv2.rectangle(frame, (original_cx1, original_cy1), (original_cx2, original_cy2), color_list[color_class_id], -1)
                        # cv2.putText(frame, f'Color Conf: {color_confidence:.2f}', 
                        #             (original_cx1, original_cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[color_class_id], 2)

        # Display the annotated frame with color predictions
        cv2.imshow('YOLOv11 Video Prediction', frame)
        # print(f"FPS: {1 / (time.time() - start_time)}")
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    capture.release()
    cv2.destroyAllWindows()
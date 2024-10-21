import cv2
import time
import numpy as np
from ultralytics import YOLO

def sliding_window(image, step_size, window_size):
    """Slide a window across the image and return a list of windows with their positions."""
    windows = []
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if window.shape[0] == window_size[1] and window.shape[1] == window_size[0]:
                windows.append((x, y, window))
    return windows

if __name__ == "__main__":
    color_list = [(255,0,0),(0,255,255),(0,255,0),(255,0,255)]
    # Load the trained color model
    model_color = YOLO("runs_color/detect/train4/weights/best.pt")  # Ensure the path is correct

    # Open the video file
    capture = cv2.VideoCapture('resource/3.mp4')
    start_frame = 45000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    window_size = (35, 35)  # Define the size of the sliding window
    step_size = 30  # Define the step size for sliding

    while True:
        start_time = time.time()
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1080, 720))

        # Get all windows and their positions
        windows = sliding_window(frame, step_size, window_size)

        # Process each window individually
        for (x, y, window) in windows:
            # Perform prediction on the current window using model_color
            color_results = model_color.predict(window)

            for color_result in color_results:
                for color_box in color_result.boxes:
                    # Extract color prediction details
                    cx1, cy1, cx2, cy2 = color_box.xyxy.tolist()[0]
                    color_confidence = float(color_box.conf)
                    color_class_id = int(color_box.cls)
                    # Translate window coordinates to original frame coordinates
                    original_cx1 = int(cx1 + x)
                    original_cy1 = int(cy1 + y)
                    original_cx2 = int(cx2 + x)
                    original_cy2 = int(cy2 + y)
                    # Draw the color prediction on the original frame
                    cv2.rectangle(frame, (original_cx1, original_cy1), (original_cx2, original_cy2), color_list[color_class_id], -1)

        # Display the annotated frame with color predictions
        cv2.imshow('YOLOv11 Video Prediction', frame)
        # print(f"FPS: {1 / (time.time() - start_time)}")
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
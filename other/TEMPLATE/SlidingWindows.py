import cv2
import numpy as np
import concurrent.futures
import time

def process_window(window):
    # 在这里执行对每个窗口的操作，比如计算均值。
    # 这里简单返回窗口的均值作为示例。
    return np.mean(window)

def sliding_window_display(image, window_size, step_size):
    img_height, img_width = image.shape[:2]
    win_height, win_width = window_size

    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        
        for y in range(0, img_height - win_height + 1, step_size):
            for x in range(0, img_width - win_width + 1, step_size):
                window = image[y:y + win_height, x:x + win_width]
                futures.append(executor.submit(process_window, window))
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    return results

if __name__ == "__main__":
    step_size = 35  # Define step size
    vedio_dir = "3.mp4"
    capture = cv2.VideoCapture(vedio_dir)
    size_percent = 100
    start_frame = 25000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = 0
    image_size = (1080, 720)
    window_size = (35, 35)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, image_size)
        start_time = time.time()
        
        # Apply the sliding window display
        results = sliding_window_display(frame, window_size, step_size)
        
        end_time = time.time()
        print(f"Processed frame in {end_time - start_time:.2f} seconds.")
        
        # Display the frame with results if necessary
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv2.destroyAllWindows()
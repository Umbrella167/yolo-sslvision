import cv2
import numpy as np

def load_templates_from_directory(directory, size_percent):
    # Dummy function to simulate loading templates
    # Replace this with actual implementation
    return []

def on_trackbar(val):
    global threshold
    threshold = val

def calculate_histogram(frame):
    hist_size = 256
    hist_range = (0, 256)
    
    b_hist = cv2.calcHist([frame], [0], None, [hist_size], hist_range)
    g_hist = cv2.calcHist([frame], [1], None, [hist_size], hist_range)
    r_hist = cv2.calcHist([frame], [2], None, [hist_size], hist_range)

    return b_hist, g_hist, r_hist

def highlight_dominant_color(frame, b_hist, g_hist, r_hist):
    # Find the most dominant color channel
    b_max = np.max(b_hist)
    g_max = np.max(g_hist)
    r_max = np.max(r_hist)
    
    # Determine which channel is the most dominant
    if b_max > g_max and b_max > r_max:
        dominant_channel = 0  # Blue
    elif g_max > b_max and g_max > r_max:
        dominant_channel = 1  # Green
    else:
        dominant_channel = 2  # Red
    
    # Create a mask to identify the dominant color
    dominant_value = np.argmax([b_hist, g_hist, r_hist], axis=1)[dominant_channel]
    mask = frame[:, :, dominant_channel] == dominant_value
    
    # Change the pixels of the dominant color to white
    frame[mask] = [255, 255, 255]
    
    return frame

if __name__ == "__main__":
    temp_dir = "resource/template"
    vedio_dir = "3.mp4"
    capture = cv2.VideoCapture(vedio_dir)
    
    size_percent = 100
    templates = load_templates_from_directory(temp_dir, size_percent)
    start_frame = 25000
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    threshold = 50  # Initial threshold value
    
    cv2.namedWindow('YOLOv11 Video Prediction')
    cv2.createTrackbar('Threshold', 'YOLOv11 Video Prediction', threshold, 100, on_trackbar)
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1080, 720))
        
        # Calculate histogram
        b_hist, g_hist, r_hist = calculate_histogram(frame)
        
        # Highlight the dominant color
        highlighted_frame = highlight_dominant_color(frame, b_hist, g_hist, r_hist)
        
        cv2.imshow('YOLOv11 Video Prediction', highlighted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
import cv2
import os

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def load_templates_from_directory(directory, scale_percent):
    templates = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        image = cv2.imread(file_path)
        image = resize_image(image, scale_percent)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is not None:
            templates.append(image)
    return templates

def match_template(frame, template, threshold):
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    locations = []
    h, w = template.shape[:2]
    loc = cv2.minMaxLoc(result)
    if loc[1] > threshold:
        locations.append(loc[3])
    return locations, w, h

def on_trackbar(val):
    global threshold
    threshold = val / 100

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
        cv2.imshow('frame', frame)

        frame = cv2.Canny(frame, 20, 100)
        
        # frame = resize_image(frame, size_percent)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # for template in templates:
        #     locations, w, h = match_template(frame, template, threshold)
        #     for pt in locations:
        #         cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        
        cv2.imshow('Canny', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()
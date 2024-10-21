from ultralytics import YOLO
import cv2
import time
from ultralytics import YOLO

def train():
    model = YOLO("resource/yolo11n.pt")  # 加载模型
    trainer = model.train(data="resource/data.yaml", epochs=45)  # 训练模型

# 调用训练函数

if __name__ == "__main__":
    # train()
    # 加载训练好的模型
    model = YOLO("runs/detect/train10/weights/best.pt")  # 请确保路径正确
    # 打开视频文件
    capture = cv2.VideoCapture('resource/3.mp4')
    
    # 获取视频的宽度、高度和帧率
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    # 定义视频编码器和创建VideoWriter对象
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用'mp4v'编码器

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # 对当前帧进行预测
        results = model.predict(frame)

        # 显示预测结果
        annotated_frame = results[0].plot()  # 绘制带注释的帧

        annotated_frame = cv2.resize(annotated_frame, (1200, 900))
        cv2.imshow('YOLOv11 Video Prediction', annotated_frame)
        # time.sleep(0.01)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象和写入器，并关闭所有OpenCV窗口
    capture.release()
    cv2.destroyAllWindows()
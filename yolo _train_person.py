from ultralytics import YOLO

# 加载预训练模型（会自动下载）
model = YOLO('yolov8n.pt')

# 开始训练
model.train(
    data='C:\\Users\\Administrator\\Desktop\\YOLO\\dataset\\data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',      # 0 表示第一块 GPU，'cpu' 表示使用 CPU
    patience=50,       # 早停耐心值
    save_period=10     # 每 10 轮保存一次检查点
)
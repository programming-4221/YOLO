from pathlib import Path
from ultralytics import YOLO

model = YOLO("best.pt")
img_dir = Path('input')                 # 图片所在目录
out_dir = Path('output'); out_dir.mkdir(exist_ok=True)

# 收集所有图片（可自行增删后缀）
img_files = [p for p in img_dir.iterdir()
             if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

for img_file in img_files:
    results = model(str(img_file), imgsz=640, conf=0.25)
    results[0].save(filename=str(out_dir / f"{img_file.stem}_result.jpg"))
    print(f"{img_file.name}: 检测到 {len(results[0].boxes)} 个目标")
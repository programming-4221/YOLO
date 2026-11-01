import cv2
import mediapipe as mp
import threading
import time
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# ================= 配置区 =================
YOLO_MODEL = "yolov8n.pt"
MEDIAPIPE_MODEL = "face_landmarker.task"
CONF_THRESHOLD = 0.5
MAX_FACES = 5  # ★ 支持同时检测的人脸数量
# ==========================================

# ---------- 共享数据与锁 ----------
frame_lock = threading.Lock()
yolo_results = None
mediapipe_landmarks = None
running = True

# ---------- 获取人脸连线数据 ----------
# MediaPipe 的人脸有 478 个点，连线非常复杂。
# 我们尝试从旧版接口获取完整的三角形网格（视觉效果最好）
try:
    import mediapipe as mp

    # 完整的面部网格连线 (包含几千条线)
    FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
except AttributeError:
    # 兼容模式：如果新版移除了上述接口，我们手动定义核心的轮廓线（脸型、眼睛、嘴唇）
    print("[提示] 未找到完整网格连线，使用手动定义的轮廓连线。")
    FACE_OVAL = [(10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
                 (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
                 (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
                 (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103),
                 (103, 67), (67, 109), (109, 10)]
    LEFT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (133, 173),
                (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246), (246, 33)]
    RIGHT_EYE = [(362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263),
                 (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)]
    LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375),
            (375, 291), (291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40),
            (40, 185), (185, 61)]
    FACE_CONNECTIONS = FACE_OVAL + LEFT_EYE + RIGHT_EYE + LIPS


def yolo_worker(cap):
    global yolo_results, running
    model = YOLO(YOLO_MODEL)
    while running:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, conf=CONF_THRESHOLD, verbose=False, stream=True)
        for r in results:
            with frame_lock:
                yolo_results = r


def mediapipe_worker(cap):
    global mediapipe_landmarks, running

    if not os.path.exists(MEDIAPIPE_MODEL):
        raise FileNotFoundError(f"找不到模型文件: {MEDIAPIPE_MODEL}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    def result_callback(result, output_image, timestamp_ms):
        global mediapipe_landmarks
        with frame_lock:
            mediapipe_landmarks = result

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
        num_faces=MAX_FACES  # 支持多人脸
    )

    start_time = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while running:
            ret, frame = cap.read()
            if not ret: break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            landmarker.detect_async(mp_image, timestamp_ms)


def main():
    global running, yolo_results, mediapipe_landmarks

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    yolo_thread = threading.Thread(target=yolo_worker, args=(cap,))
    mp_thread = threading.Thread(target=mediapipe_worker, args=(cap,))
    yolo_thread.start()
    mp_thread.start()

    print("按 'q' 或 'ESC' 退出...")

    while running:
        ret, display_frame = cap.read()
        if not ret: break

        with frame_lock:
            # 1. 绘制 YOLO 边界框
            if yolo_results is not None:
                for box in yolo_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{yolo_results.names[cls]} {conf:.2f}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. 绘制 MediaPipe 人脸网格 (点 + 线)
            if mediapipe_landmarks is not None and mediapipe_landmarks.face_landmarks:
                h, w = display_frame.shape[:2]

                # 遍历画面中的每一张脸
                for face_landmarks in mediapipe_landmarks.face_landmarks:
                    # 先将所有关键点坐标转换为像素坐标
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

                    # 第一步：画线 (网格或轮廓)
                    for connection in FACE_CONNECTIONS:
                        start_idx, end_idx = connection
                        # 确保索引不越界
                        if start_idx < len(points) and end_idx < len(points):
                            pt1 = points[start_idx]
                            pt2 = points[end_idx]
                            # 画线：颜色使用浅青色，线宽1
                            cv2.line(display_frame, pt1, pt2, (255, 200, 0), 1, cv2.LINE_AA)

                    # 第二步：画点
                    for pt in points:
                        # 画点：颜色使用红色，实心圆
                        cv2.circle(display_frame, pt, 1, (0, 0, 255), -1)

        cv2.imshow("YOLO + Face Mesh (Multi-Face)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False
            break

    yolo_thread.join()
    mp_thread.join()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
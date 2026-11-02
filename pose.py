import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ==================== 配置 ====================
MODEL_PATH = "yolo26n-pose.pt"          # 首次运行会自动下载
CONF_THRES = 0.25                        # 人体检测置信度
KP_CONF_THRES = 0.30                     # 关键点置信度阈值

# 疲劳判定阈值（可根据摄像头角度微调）
DY_NORM_THRESH = 0.28                    # 头部前倾度阈值
EAR_SHOULDER_THRESH = 0.55               # 耳-肩垂直距离 / 肩宽 的下限
WINDOW_SEC = 3.0                         # 滑动窗口时长（秒）
ABNORMAL_RATIO_THRESH = 0.6              # 窗口内异常帧占比超过此值 → 疲劳
MIN_VALID_FRAMES = 15                    # 窗口内至少需多少有效帧才判断

# COCO 关键点索引
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def get_xy_conf(keypoints_xy, keypoints_conf, idx):
    """安全获取单个关键点的 (x, y) 和置信度。"""
    if keypoints_xy is None or idx >= len(keypoints_xy):
        return None, 0.0
    x, y = keypoints_xy[idx]
    conf = float(keypoints_conf[idx]) if keypoints_conf is not None else 1.0
    return (float(x), float(y)), conf


def compute_fatigue_indicators(kxy, kconf):
    """
    计算与视角无关的疲劳指标。
    返回 dict: 包含 head_drop (Δy_norm), ear_shoulder_ratio, valid
    如果关键点不足，valid=False。
    """
    nose, c_nose = get_xy_conf(kxy, kconf, NOSE)
    l_eye, c_le = get_xy_conf(kxy, kconf, L_EYE)
    r_eye, c_re = get_xy_conf(kxy, kconf, R_EYE)
    l_ear, c_lear = get_xy_conf(kxy, kconf, L_EAR)
    r_ear, c_rear = get_xy_conf(kxy, kconf, R_EAR)
    l_sh, c_ls = get_xy_conf(kxy, kconf, L_SHOULDER)
    r_sh, c_rs = get_xy_conf(kxy, kconf, R_SHOULDER)
    l_hip, c_lh = get_xy_conf(kxy, kconf, L_HIP)
    r_hip, c_rh = get_xy_conf(kxy, kconf, R_HIP)

    # 肩部必须可见（上半身检测的最低要求）
    if c_ls < KP_CONF_THRES or c_rs < KP_CONF_THRES:
        return {"valid": False}

    shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    shoulder_width = abs(l_sh[0] - r_sh[0])
    if shoulder_width < 10:  # 画面中人体过小，不可靠
        return {"valid": False}

    # ---------- 自适应参考基准 ----------
    hip_visible = c_lh >= KP_CONF_THRES and c_rh >= KP_CONF_THRES
    if hip_visible:
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        # 用躯干高度做辅助尺度（可选，这里仍用肩宽保持尺度一致性）
        # 躯干向量可用于判断整体前倾，但疲劳主要看头颈，这里保留备用
        trunk_vector = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])

    # ---------- 指标1: 头部前倾度 Δy_norm ----------
    head_drop = None
    if c_nose >= KP_CONF_THRES:
        if c_le >= KP_CONF_THRES and c_re >= KP_CONF_THRES:
            eye_avg_y = (l_eye[1] + r_eye[1]) / 2
        elif c_lear >= KP_CONF_THRES and c_rear >= KP_CONF_THRES:
            # 眼睛不可见时，用耳部 y 近似头部高度（上半身侧脸常见）
            eye_avg_y = (l_ear[1] + r_ear[1]) / 2
        else:
            eye_avg_y = None

        if eye_avg_y is not None:
            head_drop = (nose[1] - eye_avg_y) / shoulder_width

    # ---------- 指标2: 耳-肩垂直距离 ----------
    ear_shoulder_ratio = None
    ears = []
    if c_lear >= KP_CONF_THRES:
        ears.append(l_ear)
    if c_rear >= KP_CONF_THRES:
        ears.append(r_ear)
    if len(ears) > 0:
        ear_avg_y = sum(e[1] for e in ears) / len(ears)
        ear_shoulder_ratio = (shoulder_mid[1] - ear_avg_y) / shoulder_width

    valid = (head_drop is not None) or (ear_shoulder_ratio is not None)
    if not valid:
        return {"valid": False}

    return {
        "valid": True,
        "head_drop": head_drop,
        "ear_shoulder_ratio": ear_shoulder_ratio,
        "hip_visible": hip_visible,
    }


def is_frame_abnormal(indicators):
    """单帧是否呈现疲劳姿态。任一指标超过阈值即视为异常。"""
    if not indicators.get("valid"):
        return False

    bad = False
    hd = indicators.get("head_drop")
    if hd is not None and hd > DY_NORM_THRESH:
        bad = True
    esr = indicators.get("ear_shoulder_ratio")
    if esr is not None and esr < EAR_SHOULDER_THRESH:
        bad = True
    return bad


def main():
    model = YOLO(MODEL_PATH)

    # 滑动窗口：存储 (timestamp, is_abnormal, indicators)
    window = deque()
    cap = cv2.VideoCapture(0)  # 摄像头
    # 如果用视频文件: cap = cv2.VideoCapture("input.mp4")

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    fatigue_state = False  # 持续疲劳状态

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = cv2.getTickCount() / cv2.getTickFrequency()

        results = model(frame, conf=CONF_THRES, verbose=False)
        frame = results[0].plot()  # 返回新图，必须重新赋值给 frame

        # 默认画“正常”
        status_text = "NORMAL"
        status_color = (0, 200, 0)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 取置信度最高的人体（画面中通常只有一人）
            kxy_all = results[0].keypoints.xy.cpu().numpy()
            kconf_all = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints.conf is not None else None

            # 选择检测框置信度最高的个体
            box_confs = results[0].boxes.conf.cpu().numpy()
            person_idx = int(np.argmax(box_confs)) if len(box_confs) > 0 else 0
            kxy = kxy_all[person_idx]
            kconf = kconf_all[person_idx] if kconf_all is not None else None

            indicators = compute_fatigue_indicators(kxy, kconf)
            abnormal = is_frame_abnormal(indicators)

            # 维护滑动窗口（仅保留最近 WINDOW_SEC 秒内的帧）
            window.append((now, abnormal, indicators))
            while window and now - window[0][0] > WINDOW_SEC:
                window.popleft()

            # 统计窗口内有效帧与异常帧
            valid_frames = [(t, ab, ind) for t, ab, ind in window if ind.get("valid")]
            if len(valid_frames) >= MIN_VALID_FRAMES:
                abnormal_count = sum(1 for _, ab, _ in valid_frames if ab)
                ratio = abnormal_count / len(valid_frames)
                if ratio >= ABNORMAL_RATIO_THRESH:
                    fatigue_state = True
                else:
                    fatigue_state = False
            # 如果有效帧不足，保持上一状态（避免画面切换时跳变）

            if fatigue_state:
                status_text = "FATIGUE DETECTED"
                status_color = (0, 0, 255)

            # 在画面上叠加关键指标
            if indicators.get("valid"):
                hd = indicators.get("head_drop")
                esr = indicators.get("ear_shoulder_ratio")
                info_lines = []
                if hd is not None:
                    info_lines.append(f"Head drop: {hd:.2f} (>{DY_NORM_THRESH})")
                if esr is not None:
                    info_lines.append(f"Ear-shoulder: {esr:.2f} (<{EAR_SHOULDER_THRESH})")
                info_lines.append("Ref: Hip" if indicators.get("hip_visible") else "Ref: Shoulder")
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, 30 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # 绘制状态
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.imshow("YOLOv26n_pose Fatigue Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
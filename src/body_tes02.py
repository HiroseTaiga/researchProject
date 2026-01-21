import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import collections
import socket, json

HOST = '127.0.0.1'
PORT = 50007
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("カメラが検出できません。")

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        # === 初期値 ===
        angle = 0.0
        pose_data = {"landmarks": []}

        # === 検出結果あり ===
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # ランドマーク登録
            for lm in results.pose_landmarks.landmark:
                if lm.visibility > 0.5:
                    pose_data["landmarks"].append({
                        "x": lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility
                    })
                else:
                    pose_data["landmarks"].append(None)

        # === JSON送信（None対策あり） ===
        try:
            json_str = json.dumps(pose_data)
            sock.sendto(json_str.encode('utf-8'), (HOST, PORT))
        except Exception as e:
            print("送信エラー:", e)

        cv2.imshow('MediaPipe Pose (Improved)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()

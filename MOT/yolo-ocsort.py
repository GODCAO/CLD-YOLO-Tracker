import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from common_config import *  # dataset_root, splits, trackeval_tracker_root
from ocsort.ocsort import OCSort  # 你提供的 OC-SORT 类

import time
total_frames = 0
total_time = 0.0

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [xc, yc, w, h]

def run_sort(split):
    split_dir = os.path.join(dataset_root, split)

    for seq in sorted(os.listdir(split_dir)):
        start_time = time.time()
        num_frames = 0
        seq_dir = os.path.join(split_dir, seq)
        if not os.path.isdir(seq_dir):
            continue

        # ===== 找视频 =====
        video = None
        for ext in [".mp4", ".avi", ".mov"]:
            p = os.path.join(seq_dir, seq + ext)
            if os.path.exists(p):
                video = p
                break
        if video is None:
            print(f"[SKIP] no video: {seq}")
            continue

        print(f"[RUN] YOLO + OC-SORT -> {seq}")

        # ===== TrackEval 输出路径 =====
        out_dir = os.path.join(trackeval_tracker_root, "YOLO_OCSORT", split)
        os.makedirs(out_dir, exist_ok=True)
        out_txt = os.path.join(out_dir, f"{seq}.txt")
        if os.path.exists(out_txt):
            os.remove(out_txt)

        cap = cv2.VideoCapture(video)
        assert cap.isOpened()

        # ===== 初始化 YOLOv12 =====
        model = YOLO(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # ===== 初始化 OC-SORT =====
        mot_tracker = OCSort(det_thresh=0.4, max_age=30, min_hits=3, asso_func="iou")

        frame_id = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_h, img_w = frame.shape[:2]

            # ===== YOLOv12 detection =====
            results = model.predict(
                source=frame,
                conf=0.4,
                iou=0.5,
                device=device,
                verbose=False
            )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                frame_id += 1
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            dets = np.concatenate([xyxy, confs[:, None]], axis=1)  # [x1,y1,x2,y2,score]

            # ===== 更新 OC-SORT =====
            trackers = mot_tracker.update(dets, img_info=(img_h, img_w), img_size=(img_h, img_w))

            # ===== Save MOT format =====
            if len(trackers) > 0:
                with open(out_txt, "a") as f:
                    for d in trackers:
                        # OC-SORT 默认输出格式 [x1,y1,x2,y2,ID] 或 [x1,y1,x2,y2,ID,cate,...]
                        frame_id_out = frame_id
                        tid = int(d[4])
                        x1, y1, x2, y2 = d[:4]
                        w = x2 - x1
                        h = y2 - y1
                        f.write(f"{frame_id_out},{tid},{int(x1)},{int(y1)},{int(w)},{int(h)},-1,-1,-1,-1\n")

            frame_id += 1
            num_frames += 1

        cap.release()
        seq_time = time.time() - start_time

        global total_frames, total_time
        total_frames += num_frames
        total_time += seq_time
        print(f"[OK] saved: {out_txt}")


if __name__ == "__main__":
    for split in splits:
        run_sort(split)
    overall_fps = total_frames / total_time

    print("=" * 60)
    print(f"[OVERALL FPS]")
    print(f"Total frames : {total_frames}")
    print(f"Total time   : {total_time:.2f} s")
    print(f"Overall FPS  : {overall_fps:.2f}")
    print("=" * 60)
    import csv

    csv_path = os.path.join(trackeval_tracker_root, "OCSORT_fps.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tracker", "total_frames", "total_time_sec", "overall_fps"])
        writer.writerow([
            "YOLO + OCSORT",
            total_frames,
            round(total_time, 3),
            round(overall_fps, 2)
        ])
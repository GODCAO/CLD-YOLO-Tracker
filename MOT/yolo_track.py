import os
import time
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from common_config import *

# ================== 模型 ==================
model = YOLO(model_path)

# ================== 全局统计 ==================
total_frames = 0
total_time = 0.0
total_latency_time = 0.0
global_gpu_mem_peak = 0.0


def run_track(split):
    global total_frames, total_time, total_latency_time, global_gpu_mem_peak

    split_dir = os.path.join(dataset_root, split)

    for seq in sorted(os.listdir(split_dir)):

        seq_dir = os.path.join(split_dir, seq)
        if not os.path.isdir(seq_dir):
            continue

        # ===== 找视频 =====
        video = None
        for ext in ['.mp4', '.avi', '.mov']:
            p = os.path.join(seq_dir, seq + ext)
            if os.path.exists(p):
                video = p
                break

        if video is None:
            print(f'[SKIP] no video: {seq}')
            continue

        # ===== 帧数 =====
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f'[SKIP] cannot open video: {video}')
            continue

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f'[RUN] YOLO Track -> {seq} | Frames: {num_frames}')

        # ================== GPU reset ==================
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ================== 计时 ==================
        start_time = time.time()

        # ================== YOLO track ==================
        model.track(
            source=video,
            tracker=tracker_config,
            save_txt=True,
            save_conf=True,
            project=seq_dir,
            name='track',
            exist_ok=True,
            verbose=False
        )

        seq_time = time.time() - start_time

        # ================== latency (per frame) ==================
        total_latency_time += (seq_time * 1000)

        # ================== GPU memory ==================
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
            global_gpu_mem_peak = max(global_gpu_mem_peak, peak_mem)

        # ================== global stats ==================
        total_frames += num_frames
        total_time += seq_time

        fps = num_frames / seq_time if seq_time > 0 else 0

        print(f'[DONE] {seq} | Time: {seq_time:.2f}s | FPS: {fps:.2f}')


if __name__ == '__main__':

    for split in splits:
        run_track(split)

    # ================== metrics ==================
    overall_fps = total_frames / total_time if total_time > 0 else 0
    avg_latency_ms = total_latency_time / total_frames if total_frames > 0 else 0

    print('=' * 60)
    print('[OVERALL RESULTS]')
    print(f'Total frames : {total_frames}')
    print(f'Total time   : {total_time:.2f} s')
    print(f'Overall FPS  : {overall_fps:.2f}')
    print(f'Latency      : {avg_latency_ms:.3f} ms/frame')
    print(f'GPU peak mem : {global_gpu_mem_peak:.2f} MB')
    print('=' * 60)

    # ================== CSV ==================
    output_root = "results"
    csv_path = os.path.join(output_root, 'fps_result.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Detector',
            'Tracker',
            'Total_Frames',
            'Total_Time(s)',
            'Overall_FPS',
            'Latency(ms)',
            'GPU_Memory(MB)'
        ])
        writer.writerow([
            'YOLOv12',
            Path(tracker_config).stem,
            total_frames,
            f'{total_time:.2f}',
            f'{overall_fps:.2f}',
            f'{avg_latency_ms:.3f}',
            f'{global_gpu_mem_peak:.2f}'
        ])

    print(f'[OK] FPS result saved to: {csv_path}')
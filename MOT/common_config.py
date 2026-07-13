# 路径
import os

# ===== 基本路径 =====
dataset_root = ''
# splits = ['train','val']
splits = ['val']
img_dir_name = 'img1'
img_ext = '.jpg'
frame_rate = 30

# ===== YOLO =====
# model path
model_path = (
    ''
)
# tracker_config = 'botsort.yaml'
# # bytetrack  botsort

# ===== TrackEval =====
trackeval_root = 'TrackEval'
trackeval_gt_root = os.path.join(
    trackeval_root, 'data/gt/mot_challenge/MOT_dataset'
)
trackeval_tracker_root = os.path.join(
    trackeval_root, 'data/trackers/mot_challenge/MOT_dataset'
)
>>>>>>> 76f6efd74f04d3be6a57599f533fd8cff2e0225b

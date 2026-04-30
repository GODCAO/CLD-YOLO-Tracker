# CLD-YOLO-Tracker


# Bash
conda create -n cldtrack python=3.11 \n
conda activate cldtrack \n
pip install -r requirements.txt \n


# Train object model
import warnings \n
from ultralytics.utils.torch_utils import profile \n
warnings.filterwarnings('ignore') \n
from ultralytics import YOLO, RTDETR \n
import os \n

if __name__ == '__main__': \n
  model = YOLO('CLD-YOLO.yaml') \n
  results = model.train( \n
    data='datasets path',   \n 
    epochs=200, \n
    batch=32, \n
    imgsz=640, \n
    scale=0.5, \n
    mosaic=1.0, \n
    mixup=0.0, \n
    copy_paste=0.1, \n
    device=0,  \n
    optimizer='SGD',  \n
    workers=8,\n
 ) 


# Test 
import warnings \n
warnings.filterwarnings('ignore') \n
from ultralytics import YOLO, RTDETR \n

if __name__ == "__main__": \n
    model = YOLO('model path')   # best.pt  \n
    model.val(data='datasets path', device=0, workers=0, save_json=True) \n


# Tracking(MOT) 
The parameter settings are located in the common_config.py. \n
Run yolo-ocsort.py \n

CLD-YOLO-Tracker


<>Bash
conda create -n cldtrack python=3.11
conda activate cldtrack
pip install -r requirements.txt


Train object model
import warnings
from ultralytics.utils.torch_utils import profile
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR
import os

if __name__ == '__main__':
  model = YOLO('CLD-YOLO.yaml')
  # model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml")
  # model.load('yolo12n.pt')
  results = model.train(
    data='',  #  datasets path
    epochs=200, 
    batch=32, 
    imgsz=640, 
    scale=0.5, 
    mosaic=1.0, 
    mixup=0.0, 
    copy_paste=0.1, 
    device=0,  
    optimizer='SGD',  
    workers=8,
 
)


Test 
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == "__main__":
    model = YOLO('model path')   # best.pt 
    # model = RTDETR('model path')
    model.val(data='datasets path', device=0, workers=0, save_json=True)


Tracking(MOT)
The parameter settings are located in the common_config.py.
Run yolo-ocsort.py

from ultralytics import YOLO
from PIL import Image

# 1. 加载模型
model = YOLO('D:/code/Python/yolov12/runs/detect_200epochs/detect_peachs/train/train_v12ciou_silu/weights/best.pt')

# 2. 预测并保存到指定文件夹
results = model.predict(
    source="D:/code/peachs/images/val/00153_000150.png",
    save=True,
)

# 3. 获取保存的预测图片路径
# Ultralytics 返回的 results 对象里包含保存路径
pred_img_path = results[0].plot(save=False)  # plot=False 不额外保存
saved_path = "runs/detect/predict/00153_000150.jpg"

# 4. 用 Pillow 打开并修改 DPI
img = Image.open(saved_path)
img.save("runs/detect/predict/00153_000150_pred_300dpi.jpg", dpi=(300, 300))

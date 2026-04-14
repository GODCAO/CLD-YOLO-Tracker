# 将coco格式转换为yolo格式
import os
import json
from tqdm import tqdm
import shutil


class CocoToYoloConverter:
    def __init__(self):
        self.category_map = {}
        self.seen_images = set()

    def convert(self, coco_ann_path, image_source_dir, output_root, splits=None):
        """
        主转换函数
        :param coco_ann_path: COCO标注文件路径（或包含split的路径模板）
        :param image_source_dir: 原始图片根目录
        :param output_root: 输出根目录
        :param splits: 数据集划分列表，如['train', 'val']
        """
        if splits is None:
            splits = ['train']

        # 初始化目录结构
        self._create_dirs(output_root, splits)

        # 处理每个数据划分
        for split in splits:
            # 动态处理标注文件路径
            ann_file = coco_ann_path.format(split=split) if '{split}' in coco_ann_path else coco_ann_path

            # 加载标注数据
            with open(ann_file) as f:
                data = json.load(f)

            # 构建类别映射（只在第一次处理时）
            if not self.category_map:
                self._build_category_map(data['categories'], output_root)

            # 处理当前划分的数据
            self._process_split(data, image_source_dir, output_root, split)

    def _create_dirs(self, output_root, splits):
        """创建YOLO格式所需目录"""
        os.makedirs(output_root, exist_ok=True)

        # 创建images和labels子目录
        for split in splits:
            os.makedirs(os.path.join(output_root, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_root, 'labels', split), exist_ok=True)

    def _build_category_map(self, categories, output_root):
        """构建类别ID映射并保存classes.txt"""
        # 按原始ID排序后生成连续映射
        sorted_cats = sorted(categories, key=lambda x: x['id'])
        self.category_map = {cat['id']: idx for idx, cat in enumerate(sorted_cats)}

        # 保存类别文件
        class_file = os.path.join(output_root, 'labels', 'classes.txt')
        with open(class_file, 'w') as f:
            for cat in sorted_cats:
                f.write(f"{cat['name']}\n")

    def _process_split(self, data, image_source_dir, output_root, split):
        """处理单个数据划分"""
        # 创建图像ID到信息的映射
        images = {img['id']: img for img in data['images']}

        # 准备索引文件内容
        index_content = []

        # 使用进度条处理标注
        for ann in tqdm(data['annotations'], desc=f"Processing {split} annotations"):
            # 跳过crowd标注
            if ann.get('iscrowd', 0) == 1:
                continue

            # 获取关联的图像信息
            img_info = images.get(ann['image_id'])
            if not img_info:
                continue

            # 处理图像文件
            img_path = self._process_image(img_info, image_source_dir, output_root, split)
            if img_path and img_path not in self.seen_images:
                index_content.append(img_path)
                self.seen_images.add(img_path)

            # 处理标注
            self._process_annotation(ann, img_info, output_root, split)

        # 保存索引文件
        self._save_index_file(index_content, output_root, split)

    def _process_image(self, img_info, src_dir, output_root, split):
        """处理图像文件并返回相对路径"""
        src_path = os.path.join(src_dir, img_info['file_name'])
        dst_dir = os.path.join(output_root, 'images', split)
        dst_path = os.path.join(dst_dir, img_info['file_name'])

        # 检查源文件是否存在
        if not os.path.exists(src_path):
            print(f"Warning: Missing source image {src_path}")
            return None

        # 复制图像（如果尚未复制）
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

        return os.path.relpath(dst_path, output_root)

    def _process_annotation(self, ann, img_info, output_root, split):
        """处理单个标注并保存到标签文件"""
        # 获取YOLO格式的类别ID
        yolo_cls = self.category_map.get(ann['category_id'])
        if yolo_cls is None:
            return

        # 转换边界框坐标
        try:
            x, y, w, h = ann['bbox']
            img_w, img_h = img_info['width'], img_info['height']

            # 计算归一化坐标
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # 验证坐标有效性
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < w_norm <= 1 and 0 < h_norm <= 1):
                return
        except (KeyError, TypeError, ZeroDivisionError) as e:
            print(f"Invalid bbox in image {img_info['id']}: {e}")
            return

        # 准备标签行
        label_line = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        # 写入标签文件
        label_dir = os.path.join(output_root, 'labels', split)
        txt_name = os.path.splitext(img_info['file_name'])[0] + '.txt'
        txt_path = os.path.join(label_dir, txt_name)

        with open(txt_path, 'a') as f:
            f.write(label_line + '\n')

    def _save_index_file(self, content, output_root, split):
        """保存索引文件（train.txt/val.txt）"""
        index_file = os.path.join(output_root, f"{split}.txt")
        with open(index_file, 'w') as f:
            for path in content:
                f.write(f"datasets/logo/{path}\n")


if __name__ == '__main__':
    # 使用示例
    converter = CocoToYoloConverter()

    # 假设标注文件路径包含{split}占位符
    converter.convert(
        coco_ann_path="D:/code/datasets/orange_detection/annotations/instances_train.json",
        image_source_dir="D:/code/datasets/orange_detection/train",
        output_root="datasets/orange_detection",
        splits=['train']
    )
    converter.convert(
        coco_ann_path="D:/code/datasets/orange_detection/annotations/instances_val.json",
        image_source_dir="D:/code/datasets/orange_detection/val",
        output_root="datasets/orange_detection",
        splits=['val']
    )
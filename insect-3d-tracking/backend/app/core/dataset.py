import os
import shutil
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import random
from sklearn.model_selection import train_test_split

from ..config import settings


class DatasetManager:
    """数据集管理器类"""
    
    def __init__(self, dataset_name: str, base_dir: str = settings.DATASET_DIR):
        """
        初始化数据集管理器
        
        Args:
            dataset_name: 数据集名称
            base_dir: 数据集存储的基础目录
        """
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir)
        self.dataset_path = self.base_dir / self.dataset_name
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        self.annotations_path = self.dataset_path / "annotations"
        self.config_path = self.dataset_path / "dataset.yaml"
        
        self._create_dirs()
    
    def _create_dirs(self):
        """创建数据集所需目录"""
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.annotations_path.mkdir(parents=True, exist_ok=True)
    
    def add_image(self, image_path: str, annotation: Optional[Dict] = None) -> str:
        """
        向数据集中添加单个图像和可选的标注
        
        Args:
            image_path: 原始图像路径
            annotation: 图像的标注数据 (COCO格式)
            
        Returns:
            str: 新图像在数据集中的路径
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 复制图像
        image_filename = Path(image_path).name
        new_image_path = self.images_path / image_filename
        shutil.copy(image_path, new_image_path)
        
        # 如果有标注，则保存
        if annotation:
            annotation_filename = new_image_path.stem + ".json"
            annotation_path = self.annotations_path / annotation_filename
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=4)
        
        return str(new_image_path)
    
    def add_video_frames(
        self, 
        video_path: str, 
        frame_indices: Optional[List[int]] = None,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        从视频中提取帧并添加到数据集
        
        Args:
            video_path: 视频文件路径
            frame_indices: 要提取的特定帧的索引列表
            frame_interval: 每隔多少帧提取一帧
            max_frames: 最多提取的帧数
            
        Returns:
            List[str]: 添加的图像路径列表
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")
        
        added_images = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = Path(video_path).stem
        
        if frame_indices:
            # 提取指定帧
            for i in frame_indices:
                if i < frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        image_filename = f"{video_name}_frame_{i}.jpg"
                        image_path = self.images_path / image_filename
                        cv2.imwrite(str(image_path), frame)
                        added_images.append(str(image_path))
        elif frame_interval:
            # 按间隔提取帧
            count = 0
            for i in range(0, frame_count, frame_interval):
                if max_frames and count >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    image_filename = f"{video_name}_frame_{i}.jpg"
                    image_path = self.images_path / image_filename
                    cv2.imwrite(str(image_path), frame)
                    added_images.append(str(image_path))
                    count += 1
        
        cap.release()
        return added_images
    
    def save_annotation(self, image_name: str, annotation: Dict):
        """
        保存单个图像的标注
        
        Args:
            image_name: 图像文件名 (e.g., 'image1.jpg')
            annotation: COCO格式的标注数据
        """
        annotation_filename = Path(image_name).stem + ".json"
        annotation_path = self.annotations_path / annotation_filename
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)
    
    def convert_to_yolo(self):
        """
        将COCO格式的标注转换为YOLO格式的标签
        """
        for ann_file in self.annotations_path.glob("*.json"):
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            image_info = coco_data['images'][0]
            img_w, img_h = image_info['width'], image_info['height']
            
            yolo_lines = []
            for ann in coco_data['annotations']:
                category_id = ann['category_id']
                bbox = ann['bbox'] # [x, y, width, height]
                
                # 转换为YOLO格式 (归一化的中心点x, y, 宽度, 高度)
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h
                
                yolo_lines.append(f"{category_id} {x_center} {y_center} {w} {h}")
            
            # 保存YOLO标签文件
            label_filename = ann_file.stem + ".txt"
            label_path = self.labels_path / label_filename
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
    
    def create_yolo_config(
        self, 
        class_names: List[str], 
        train_ratio: float = 0.8,
        val_ratio: float = 0.2
    ):
        """
        创建YOLOv8所需的dataset.yaml文件
        
        Args:
            class_names: 类别名称列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        # 获取所有图像文件
        all_images = [str(p) for p in self.images_path.glob("*.jpg")]
        
        # 划分训练集和验证集
        train_images, val_images = train_test_split(
            all_images, train_size=train_ratio, random_state=42
        )
        
        # 创建配置文件内容
        config_data = {
            'path': str(self.dataset_path.resolve()),
            'train': 'images', # YOLO会自动在path下查找
            'val': 'images',   # 简单起见，可以指向同一个目录，YOLO会根据txt文件区分
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        # 创建指向训练和验证图像的txt文件
        with open(self.dataset_path / 'train.txt', 'w') as f:
            f.write('\n'.join(train_images))
        
        with open(self.dataset_path / 'val.txt', 'w') as f:
            f.write('\n'.join(val_images))
            
        # 写入yaml文件
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取数据集的统计信息
        """
        num_images = len(list(self.images_path.glob("*")))
        num_annotations = len(list(self.annotations_path.glob("*")))
        num_labels = len(list(self.labels_path.glob("*")))
        
        config = {}
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        return {
            "dataset_name": self.dataset_name,
            "path": str(self.dataset_path),
            "num_images": num_images,
            "num_annotations": num_annotations,
            "num_yolo_labels": num_labels,
            "config": config
        }

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import time
from pathlib import Path
import torch
from ultralytics import YOLO

from .apriltag import triangulate_points


class InsectDetector:
    """昆虫检测器类，使用YOLO模型进行目标检测"""
    
    def __init__(
        self, 
        model_path: str = None, 
        confidence: float = 0.25,
        device: str = None
    ):
        """
        初始化昆虫检测器
        
        Args:
            model_path: YOLO模型路径，如果为None则使用默认的YOLOv8n模型
            confidence: 检测置信度阈值
            device: 运行设备 ('cpu', 'cuda:0', 等)，如果为None则自动选择
        """
        self.confidence = confidence
        
        # 如果未指定设备，则检查是否有可用的CUDA设备
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        if model_path is None or not os.path.exists(model_path):
            # 使用默认模型
            self.model = YOLO('yolov8n.pt')
        else:
            # 使用自定义模型
            self.model = YOLO(model_path)
        
        # 强制模型使用指定设备
        self.model.to(self.device)
    
    def detect(
        self, 
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        在图像中检测昆虫
        
        Args:
            image: 输入图像
            classes: 要检测的类别ID列表，如果为None则检测所有类别
            verbose: 是否打印额外信息
            
        Returns:
            List[Dict]: 检测结果列表，每个结果包含边界框、置信度和类别信息
        """
        # 执行推理
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            classes=classes,
            verbose=verbose
        )
        
        detections = []
        
        # 解析结果
        for result in results:
            # 遍历每个检测框
            if result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for i in range(len(boxes)):
                    # YOLO输出的边界框是xyxy格式(左上和右下坐标)
                    box = boxes.xyxy[i] if hasattr(boxes, 'xyxy') else boxes.boxes[i][:4]
                    confidence = boxes.conf[i] if hasattr(boxes, 'conf') else boxes.boxes[i][4]
                    class_id = int(boxes.cls[i]) if hasattr(boxes, 'cls') else int(boxes.boxes[i][5])
                    
                    # 获取类别名称
                    class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                    
                    # 计算中心点
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # 计算宽度和高度
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    
                    detection = {
                        "xyxy": box.tolist(),                    # [x1, y1, x2, y2]
                        "center": [float(center_x), float(center_y)],  # [cx, cy]
                        "wh": [float(width), float(height)],           # [w, h]
                        "confidence": float(confidence),
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def draw_detections(
        self, 
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            color: BGR颜色
            thickness: 线条粗细
            
        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        result = image.copy()
        
        for det in detections:
            # 绘制边界框
            x1, y1, x2, y2 = map(int, det["xyxy"])
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                result,
                (x1, y1 - label_h - baseline),
                (x1 + label_w, y1),
                color,
                -1
            )
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # 绘制中心点
            center_x, center_y = map(int, det["center"])
            cv2.circle(result, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return result


class Localizer3D:
    """使用双目视觉进行3D定位的类"""
    
    def __init__(
        self, 
        camera_matrix1: np.ndarray,
        dist_coeffs1: np.ndarray,
        camera_matrix2: np.ndarray,
        dist_coeffs2: np.ndarray,
        R: np.ndarray,
        T: np.ndarray
    ):
        """
        初始化3D定位器
        
        Args:
            camera_matrix1: 第一个相机的内参矩阵
            dist_coeffs1: 第一个相机的畸变系数
            camera_matrix2: 第二个相机的内参矩阵
            dist_coeffs2: 第二个相机的畸变系数
            R: 从第一个相机坐标系到第二个相机坐标系的旋转矩阵
            T: 从第一个相机坐标系到第二个相机坐标系的平移向量
        """
        self.camera_matrix1 = camera_matrix1
        self.dist_coeffs1 = dist_coeffs1
        self.camera_matrix2 = camera_matrix2
        self.dist_coeffs2 = dist_coeffs2
        self.R = R
        self.T = T
        
        # 计算三角测量所需的投影矩阵
        self.P1 = self.camera_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P2 = self.camera_matrix2 @ np.hstack((self.R, self.T))
    
    def undistort_points(
        self, 
        points: List[Tuple[float, float]],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        对点进行去畸变处理
        
        Args:
            points: 需要去畸变的点列表
            camera_matrix: 相机内参矩阵
            dist_coeffs: 相机畸变系数
            
        Returns:
            List[Tuple[float, float]]: 去畸变后的点列表
        """
        if not points:
            return []
        
        points_array = np.array(points, dtype=np.float32)
        points_array = points_array.reshape(-1, 1, 2)
        
        # 去畸变
        undistorted_points = cv2.undistortPoints(
            points_array,
            camera_matrix,
            dist_coeffs,
            P=camera_matrix
        )
        
        # 转换回列表格式
        undistorted_points = undistorted_points.reshape(-1, 2).tolist()
        return [(p[0], p[1]) for p in undistorted_points]
    
    def triangulate(
        self, 
        points1: List[Tuple[float, float]],
        points2: List[Tuple[float, float]]
    ) -> List[Tuple[float, float, float]]:
        """
        对应点三角测量
        
        Args:
            points1: 第一个相机中的2D点
            points2: 第二个相机中的2D点，与points1一一对应
            
        Returns:
            List[Tuple[float, float, float]]: 三角测量得到的3D点
        """
        if not points1 or not points2 or len(points1) != len(points2):
            return []
        
        # 对点进行去畸变
        undistorted_points1 = self.undistort_points(
            points1, self.camera_matrix1, self.dist_coeffs1
        )
        undistorted_points2 = self.undistort_points(
            points2, self.camera_matrix2, self.dist_coeffs2
        )
        
        # 将点列表转换为NumPy数组
        points1_np = np.array(undistorted_points1, dtype=np.float32).T
        points2_np = np.array(undistorted_points2, dtype=np.float32).T
        
        # 三角测量
        points_4d = cv2.triangulatePoints(self.P1, self.P2, points1_np, points2_np)
        
        # 将结果转换为3D点列表
        points_3d = []
        for i in range(points_4d.shape[1]):
            x = points_4d[0, i] / points_4d[3, i]
            y = points_4d[1, i] / points_4d[3, i]
            z = points_4d[2, i] / points_4d[3, i]
            points_3d.append((float(x), float(y), float(z)))
        
        return points_3d
    
    def match_detections(
        self, 
        detections1: List[Dict[str, Any]],
        detections2: List[Dict[str, Any]],
        max_distance: float = 100.0,
        min_confidence: float = 0.3
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        匹配两个相机中的检测结果
        
        Args:
            detections1: 第一个相机的检测结果
            detections2: 第二个相机的检测结果
            max_distance: 对极线匹配的最大距离(像素)
            min_confidence: 最小置信度阈值
            
        Returns:
            List[Tuple[Dict, Dict]]: 匹配的检测结果对
        """
        # 筛选置信度高的检测结果
        dets1 = [d for d in detections1 if d["confidence"] >= min_confidence]
        dets2 = [d for d in detections2 if d["confidence"] >= min_confidence]
        
        # 计算基础矩阵(用于对极几何约束)
        F, _ = cv2.findFundamentalMat(
            np.array([d["center"] for d in dets1], dtype=np.float32),
            np.array([d["center"] for d in dets2], dtype=np.float32),
            cv2.FM_LMEDS
        )
        
        matches = []
        
        # 对于第一个相机中的每个检测
        for det1 in dets1:
            best_match = None
            min_epipolar_distance = float('inf')
            
            # 找出第二个相机中的最佳匹配
            for det2 in dets2:
                # 仅匹配相同类别
                if det1["class_id"] != det2["class_id"]:
                    continue
                
                # 计算对极线距离
                pt1 = np.array([det1["center"][0], det1["center"][1], 1.0])
                pt2 = np.array([det2["center"][0], det2["center"][1], 1.0])
                
                # 计算点到对极线的距离
                line = F @ pt1
                dist = abs(np.dot(line, pt2)) / np.sqrt(line[0]**2 + line[1]**2)
                
                if dist < max_distance and dist < min_epipolar_distance:
                    min_epipolar_distance = dist
                    best_match = det2
            
            if best_match:
                matches.append((det1, best_match))
        
        return matches
    
    def localize_detections(
        self, 
        detections1: List[Dict[str, Any]],
        detections2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对两个相机中匹配的检测结果进行3D定位
        
        Args:
            detections1: 第一个相机的检测结果
            detections2: 第二个相机的检测结果
            
        Returns:
            List[Dict]: 3D定位结果列表
        """
        # 匹配检测结果
        matches = self.match_detections(detections1, detections2)
        
        # 提取匹配点的中心坐标
        points1 = [match[0]["center"] for match in matches]
        points2 = [match[1]["center"] for match in matches]
        
        # 三角测量
        points_3d = self.triangulate(points1, points2)
        
        # 组合结果
        results = []
        for i, (point_3d, (det1, det2)) in enumerate(zip(points_3d, matches)):
            result = {
                "position_3d": point_3d,
                "detection1": det1,
                "detection2": det2,
                "class_id": det1["class_id"],
                "class_name": det1["class_name"],
                "confidence": (det1["confidence"] + det2["confidence"]) / 2.0
            }
            results.append(result)
        
        return results

    def localize_3d(
        self, 
        detections1: List[Dict[str, Any]],
        detections2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对两个相机中匹配的检测结果进行3D定位。
        这是一个兼容旧测试的别名，实际调用 localize_detections。
        """
        return self.localize_detections(detections1, detections2)

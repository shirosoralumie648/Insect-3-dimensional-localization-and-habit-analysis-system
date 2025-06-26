import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import math
from dataclasses import dataclass


@dataclass
class ApriltagDetection:
    """Apriltag检测结果数据类"""
    tag_id: int
    center: Tuple[float, float]  # (x, y)
    corners: List[Tuple[float, float]]  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    hamming: int
    decision_margin: float
    homography: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """将检测结果转换为字典形式"""
        return {
            "tag_id": self.tag_id,
            "center": self.center,
            "corners": self.corners,
            "hamming": self.hamming,
            "decision_margin": self.decision_margin,
            "homography": self.homography.tolist() if self.homography is not None else None
        }


class ApriltagDetector:
    """Apriltag检测器类"""
    
    def __init__(
        self, 
        family: str = "tag36h11", 
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: bool = True,
        decode_sharpening: float = 0.25,
        debug: bool = False
    ):
        """
        初始化Apriltag检测器
        
        Args:
            family: Apriltag家族类型，如"tag36h11", "tag25h9", "tag16h5", "tagCircle21h7", "tagCircle49h12"
            nthreads: 使用的线程数量
            quad_decimate: 图像预处理时的缩小因子
            quad_sigma: 应用于图像的高斯模糊sigma值
            refine_edges: 是否优化边缘
            decode_sharpening: 解码优化参数
            debug: 是否启用调试模式
        """
        self.family = family
        self.nthreads = nthreads
        self.quad_decimate = quad_decimate
        self.quad_sigma = quad_sigma
        self.refine_edges = refine_edges
        self.decode_sharpening = decode_sharpening
        self.debug = debug
        
        # 创建AprilTag检测器
        try:
            # 尝试使用pupil apriltag库
            import pupil_apriltags as apriltag
            self.detector = apriltag.Detector(
                families=family,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                quad_sigma=quad_sigma,
                refine_edges=refine_edges,
                decode_sharpening=decode_sharpening,
                debug=debug
            )
            self.backend = "pupil_apriltags"
        except ImportError:
            try:
                # 尝试使用OpenCV的contrib模块中的apriltag
                self.detector = cv2.aruco.DetectorParameters()
                if family == "tag36h11":
                    self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
                elif family == "tag25h9":
                    self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
                elif family == "tag16h5":
                    self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
                else:
                    raise ValueError(f"Unsupported tag family {family} for OpenCV backend")
                
                self.detector = cv2.aruco.ArucoDetector(self.dictionary)
                self.backend = "opencv"
            except Exception as e:
                raise ImportError(f"无法加载apriltag检测库: {str(e)}. 请安装pupil_apriltags或opencv-contrib-python。")
    
    def detect(self, image: np.ndarray) -> List[ApriltagDetection]:
        """
        在图像中检测Apriltag标签
        
        Args:
            image: 输入图像 (灰度或彩色)
            
        Returns:
            List[ApriltagDetection]: 检测到的标签列表
        """
        # 确保图像是灰度格式
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        detections = []
        
        if self.backend == "pupil_apriltags":
            # 使用pupil_apriltags检测
            tags = self.detector.detect(gray)
            
            for tag in tags:
                detection = ApriltagDetection(
                    tag_id=tag.tag_id,
                    center=(tag.center[0], tag.center[1]),
                    corners=[(p[0], p[1]) for p in tag.corners],
                    hamming=tag.hamming,
                    decision_margin=tag.decision_margin,
                    homography=tag.homography
                )
                detections.append(detection)
                
        elif self.backend == "opencv":
            # 使用OpenCV检测
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if ids is not None:
                for i, corner in enumerate(corners):
                    tag_id = int(ids[i][0])
                    tag_corners = [(corner[0][j][0], corner[0][j][1]) for j in range(4)]
                    
                    # 计算中心点
                    center_x = sum(x for x, _ in tag_corners) / 4
                    center_y = sum(y for _, y in tag_corners) / 4
                    
                    # 使用OpenCV计算单应性矩阵
                    src_pts = np.array(tag_corners, dtype=np.float32)
                    dst_pts = np.array([
                        [-1, -1], [1, -1], [1, 1], [-1, 1]
                    ], dtype=np.float32)
                    homography, _ = cv2.findHomography(src_pts, dst_pts)
                    
                    detection = ApriltagDetection(
                        tag_id=tag_id,
                        center=(center_x, center_y),
                        corners=tag_corners,
                        hamming=0,  # OpenCV不提供这个信息
                        decision_margin=1.0,  # OpenCV不提供这个信息
                        homography=homography
                    )
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[ApriltagDetection]) -> np.ndarray:
        """
        在图像上绘制检测到的Apriltag标签
        
        Args:
            image: 输入图像
            detections: 检测到的标签列表
            
        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        result = image.copy()
        
        for detection in detections:
            # 绘制边缘
            corners = np.array(detection.corners, dtype=np.int32)
            cv2.polylines(result, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            
            # 绘制中心点
            center = tuple(map(int, detection.center))
            cv2.circle(result, center, 5, (0, 0, 255), -1)
            
            # 绘制ID
            cv2.putText(
                result,
                f"ID: {detection.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return result


def calibrate_camera(
    image_points: List[List[Tuple[float, float]]],
    object_points: List[List[Tuple[float, float, float]]],
    image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    使用已知的3D-2D点对应关系标定相机
    
    Args:
        image_points: 图像中的2D点列表(每个元素是一组点)
        object_points: 对应的3D点列表(每个元素是一组点)
        image_size: 图像尺寸 (width, height)
        
    Returns:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        reprojection_error: 重投影误差
    """
    # 转换为NumPy数组
    object_points_np = [np.array(points, dtype=np.float32) for points in object_points]
    image_points_np = [np.array(points, dtype=np.float32) for points in image_points]
    
    # 初始相机矩阵猜测
    camera_matrix = np.array([
        [image_size[0], 0, image_size[0]/2],
        [0, image_size[0], image_size[1]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    
    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_np,
        image_points_np,
        (image_size[0], image_size[1]),
        camera_matrix,
        dist_coeffs,
        flags=cv2.CALIB_RATIONAL_MODEL
    )
    
    # 计算重投影误差
    total_error = 0
    for i in range(len(object_points_np)):
        imgpoints2, _ = cv2.projectPoints(
            object_points_np[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(image_points_np[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    reprojection_error = total_error / len(object_points_np)
    
    return camera_matrix, dist_coeffs, reprojection_error


def estimate_pose(
    object_points: List[Tuple[float, float, float]],
    image_points: List[Tuple[float, float]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计标签的姿态(旋转和平移)
    
    Args:
        object_points: 3D点列表
        image_points: 对应的2D点列表
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        
    Returns:
        rvec: 旋转向量
        tvec: 平移向量
    """
    # 转换为NumPy数组
    object_points_np = np.array(object_points, dtype=np.float32)
    image_points_np = np.array(image_points, dtype=np.float32)
    
    # 估计姿态
    _, rvec, tvec = cv2.solvePnP(
        object_points_np,
        image_points_np,
        camera_matrix,
        dist_coeffs
    )
    
    return rvec, tvec


def compute_relative_pose(
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个相机之间的相对姿态
    
    Args:
        rvec1: 第一个相机的旋转向量
        tvec1: 第一个相机的平移向量
        rvec2: 第二个相机的旋转向量
        tvec2: 第二个相机的平移向量
        
    Returns:
        R_rel: 相对旋转矩阵
        t_rel: 相对平移向量
    """
    # 将旋转向量转换为旋转矩阵
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    
    # 计算相对旋转和平移
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1
    
    return R_rel, t_rel


def triangulate_points(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    camera_matrix1: np.ndarray,
    camera_matrix2: np.ndarray,
    dist_coeffs1: np.ndarray,
    dist_coeffs2: np.ndarray,
    R_rel: np.ndarray,
    t_rel: np.ndarray
) -> List[Tuple[float, float, float]]:
    """
    通过双目三角测量重建3D点
    
    Args:
        points1: 第一个相机中的2D点列表
        points2: 第二个相机中对应的2D点列表
        camera_matrix1: 第一个相机的内参矩阵
        camera_matrix2: 第二个相机的内参矩阵
        dist_coeffs1: 第一个相机的畸变系数
        dist_coeffs2: 第二个相机的畸变系数
        R_rel: 相对旋转矩阵
        t_rel: 相对平移向量
        
    Returns:
        List[Tuple[float, float, float]]: 重建的3D点列表
    """
    # 转换为NumPy数组
    points1_np = np.array(points1, dtype=np.float32)
    points2_np = np.array(points2, dtype=np.float32)
    
    # 构建投影矩阵
    P1 = camera_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = camera_matrix2 @ np.hstack((R_rel, t_rel))
    
    # 执行三角测量
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, points1_np.T, points2_np.T)
    
    # 将齐次坐标转换为3D点
    points_3d = []
    for i in range(points_4d_homogeneous.shape[1]):
        x = points_4d_homogeneous[0, i] / points_4d_homogeneous[3, i]
        y = points_4d_homogeneous[1, i] / points_4d_homogeneous[3, i]
        z = points_4d_homogeneous[2, i] / points_4d_homogeneous[3, i]
        points_3d.append((float(x), float(y), float(z)))
    
    return points_3d

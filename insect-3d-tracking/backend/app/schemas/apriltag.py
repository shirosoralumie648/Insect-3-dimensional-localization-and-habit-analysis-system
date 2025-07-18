from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ApriltagConfigBase(BaseModel):
    """Apriltag配置基础模型"""
    tag_family: str  # 如"tag36h11"
    tag_id: int
    tag_size: float  # 实际大小(米)
    detection_config: Optional[Dict[str, Any]] = None


class ApriltagConfigCreate(ApriltagConfigBase):
    """Apriltag配置创建模型"""
    project_id: int


class ApriltagConfigUpdate(BaseModel):
    """Apriltag配置更新模型"""
    tag_family: Optional[str] = None
    tag_id: Optional[int] = None
    tag_size: Optional[float] = None
    detection_config: Optional[Dict[str, Any]] = None


class ApriltagConfigInDB(ApriltagConfigBase):
    """数据库中的Apriltag配置模型"""
    id: int
    project_id: int

    class Config:
        from_attributes = True


class ApriltagConfig(ApriltagConfigInDB):
    """Apriltag配置响应模型"""
    pass


class ApriltagConfigList(BaseModel):
    """Apriltag配置列表响应模型"""
    total: int
    items: List[ApriltagConfig]


class ApriltagDetection(BaseModel):
    """Apriltag检测结果模型"""
    tag_family: str
    tag_id: int
    center: Dict[str, float]  # {x, y} 中心点坐标
    corners: List[Dict[str, float]]  # [{x, y}, ...] 四个角点坐标
    pose: Optional[Dict[str, Any]] = None  # 位姿估计结果


class CalibrationResult(BaseModel):
    """相机标定结果模型"""
    camera_index: int
    camera_matrix: Dict[str, Any]  # 相机内参矩阵
    distortion_coeffs: Dict[str, Any]  # 畸变系数
    reprojection_error: float  # 重投影误差
    success: bool


class StereoCalibrationType(BaseModel):
    """双目标定类型"""
    type: str = "relative"  # "relative" 或 "absolute"
    reference_tag_id: Optional[int] = None  # 参考Apriltag ID


class StereoCalibrationResult(BaseModel):
    """双目标定结果模型"""
    left_camera_index: int
    right_camera_index: int
    rotation_matrix: Dict[str, Any]  # 旋转矩阵
    translation_vector: Dict[str, Any]  # 平移向量
    essential_matrix: Optional[Dict[str, Any]] = None  # 本质矩阵
    fundamental_matrix: Optional[Dict[str, Any]] = None  # 基础矩阵
    success: bool


class ApriltagDetectionRequest(BaseModel):
    image_base64: str
    settings: 'ApriltagDetectionSettings'

class ApriltagDetectionSettings(BaseModel):
    """Apriltag检测设置"""
    apriltag_config_id: int
    estimate_pose: bool = False
    camera_config_id: Optional[int] = None

    # Detector settings
    nthreads: int = Field(1, description="Number of threads to use")
    quad_decimate: float = Field(2.0, description="Decimate input image by this factor")
    quad_sigma: float = Field(0.0, description="Apply Gaussian blur to quad detection")
    refine_edges: bool = Field(True, description="Refine edges of quads")
    decode_sharpening: float = Field(0.25, description="Sharpening for decoding")
    debug: bool = Field(False, description="Enable debug output")
    draw_detections: bool = Field(False, description="Draw detections on the image and return it")


class ApriltagDetectionResult(BaseModel):
    """Apriltag检测的完整结果"""
    detections: List[ApriltagDetection]
    timestamp: datetime


class CalibrationData(BaseModel):
    """用于标定的数据点"""
    image_points: List[List[List[float]]]
    object_points: List[List[List[float]]]
    image_size: tuple[int, int]


class CalibrationSettings(BaseModel):
    """相机标定设置"""
    camera_config_id: int
    apriltag_config_id: int
    num_images: int = Field(20, gt=0, description="用于标定的图像数量")
    min_images: int = Field(10, gt=0, description="成功标定所需的最少图像数量")
    capture_delay: float = Field(0.5, gt=0, description="图像采集间隔(秒)")

    # Detector settings
    nthreads: int = Field(1, description="Number of threads to use")
    quad_decimate: float = Field(2.0, description="Decimate input image by this factor")
    quad_sigma: float = Field(0.0, description="Apply Gaussian blur to quad detection")
    refine_edges: bool = Field(True, description="Refine edges of quads")
    decode_sharpening: float = Field(0.25, description="Sharpening for decoding")
    debug: bool = Field(False, description="Enable debug output")


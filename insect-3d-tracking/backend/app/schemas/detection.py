from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DetectionSessionBase(BaseModel):
    """检测会话基础模型"""
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class DetectionSessionCreate(DetectionSessionBase):
    """检测会话创建模型"""
    project_id: int


class DetectionSessionUpdate(BaseModel):
    """检测会话更新模型"""
    name: Optional[str] = None
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    end_time: Optional[datetime] = None


class DetectionSessionInDB(DetectionSessionBase):
    """数据库中的检测会话模型"""
    id: int
    project_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str  # "running", "completed", "error"

    class Config:
        orm_mode = True


class DetectionSession(DetectionSessionInDB):
    """检测会话响应模型"""
    pass


class DetectionSessionList(BaseModel):
    """检测会话列表响应模型"""
    total: int
    items: List[DetectionSession]


class DetectionResultBase(BaseModel):
    """检测结果基础模型"""
    camera_index: int
    frame_number: int
    timestamp: float
    object_id: Optional[int] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center_x: float
    center_y: float


class DetectionResultCreate(DetectionResultBase):
    """检测结果创建模型"""
    session_id: int


class DetectionResultInDB(DetectionResultBase):
    """数据库中的检测结果模型"""
    id: int
    session_id: int

    class Config:
        orm_mode = True


class DetectionResult(DetectionResultInDB):
    """检测结果响应模型"""
    pass


class DetectionResultList(BaseModel):
    """检测结果列表响应模型"""
    total: int
    items: List[DetectionResult]


class TrajectoryBase(BaseModel):
    """轨迹基础模型"""
    object_id: int
    frame_number: int
    timestamp: float
    x: float
    y: float
    z: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    behavior_label: Optional[str] = None


class TrajectoryCreate(TrajectoryBase):
    """轨迹创建模型"""
    session_id: int


class TrajectoryInDB(TrajectoryBase):
    """数据库中的轨迹模型"""
    id: int
    session_id: int

    class Config:
        orm_mode = True


class Trajectory(TrajectoryInDB):
    """轨迹响应模型"""
    pass


class TrajectoryList(BaseModel):
    """轨迹列表响应模型"""
    total: int
    items: List[Trajectory]


class YoloDetectionSettings(BaseModel):
    """YOLO检测设置"""
    model_path: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: Optional[List[int]] = None
    max_det: int = 300


class TriangulationSettings(BaseModel):
    """三角测量设置"""
    min_confidence: float = 0.5  # 最小置信度
    max_reproj_error: float = 10.0  # 最大重投影误差
    tracking_algorithm: str = "sort"  # 跟踪算法: "sort", "deep_sort", "byte_track"

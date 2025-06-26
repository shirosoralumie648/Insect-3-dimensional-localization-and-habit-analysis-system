from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DetectionSessionBase(BaseModel):
    """检测会话基础模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class DetectionSessionCreate(DetectionSessionBase):
    """检测会话创建模型"""
    project_id: int


class DetectionSessionUpdate(BaseModel):
    """检测会话更新模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    end_time: Optional[datetime] = None


class DetectionSessionInDB(DetectionSessionBase):
    """数据库中的检测会话模型"""
    id: int
    project_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    created_by: int

    class Config:
        orm_mode = True


class DetectionSessionResponse(DetectionSessionInDB):
    """检测会话响应模型"""
    pass


class DetectionSessionList(BaseModel):
    """检测会话列表响应模型"""
    total: int
    items: List[DetectionSessionResponse]


class DetectionResultBase(BaseModel):
    """检测结果基础模型"""
    class_id: int
    class_name: str
    confidence: float
    box: List[float] # [x_min, y_min, x_max, y_max]
    timestamp: datetime
    frame_number: Optional[int] = None
    camera_index: Optional[int] = None
    object_id: Optional[int] = None # For tracking


class DetectionResultCreate(DetectionResultBase):
    """检测结果创建模型"""
    session_id: int


class DetectionResultUpdate(BaseModel):
    """检测结果更新模型"""
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    box: Optional[List[float]] = None
    object_id: Optional[int] = None


class DetectionResultInDB(DetectionResultBase):
    """数据库中的检测结果模型"""
    id: int
    session_id: int

    class Config:
        orm_mode = True


class DetectionResultResponse(DetectionResultInDB):
    """检测结果响应模型"""
    pass


class DetectionResultList(BaseModel):
    """检测结果列表响应模型"""
    total: int
    items: List[DetectionResultResponse]


class TrackingResultBase(BaseModel):
    """3D跟踪结果基础模型"""
    position_3d: List[float]
    confidence: float
    class_id: int
    class_name: str
    timestamp: datetime


class TrackingResultCreate(TrackingResultBase):
    """3D跟踪结果创建模型"""
    session_id: int
    detection_id1: Optional[int] = None
    detection_id2: Optional[int] = None
    tracking_data: Optional[Dict[str, Any]] = None


class TrackingResultUpdate(BaseModel):
    """3D跟踪结果更新模型"""
    position_3d: Optional[List[float]] = None
    confidence: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None


class TrackingResultInDB(TrackingResultBase):
    """数据库中的3D跟踪结果模型"""
    id: int
    session_id: int

    class Config:
        orm_mode = True


class TrackingResultResponse(TrackingResultInDB):
    """3D跟踪结果响应模型"""
    pass


class TrackingResultList(BaseModel):
    """3D跟踪结果列表响应模型"""
    total: int
    items: List[TrackingResultResponse]


class DetectionSettings(BaseModel):
    """单次检测设置"""
    model_path: str
    confidence: float = 0.25
    device: str = "cpu"
    classes: Optional[List[int]] = None


class DetectionRequest(BaseModel):
    """检测请求体"""
    session_id: Optional[int] = None
    settings: DetectionSettings
    save_result: bool = False


class LocalizationSettings(BaseModel):
    """3D定位设置"""
    session_id: Optional[int] = None
    camera_config_id1: int
    camera_config_id2: int
    model_path: str
    confidence: float = 0.25
    device: str = "cpu"
    classes: Optional[List[int]] = None
    draw_detections: bool = False


class LocalizationResult(BaseModel):
    """3D定位结果"""
    localization_results: List[Dict[str, Any]]
    count: int
    result_image1: Optional[str] = None # base64 encoded image
    result_image2: Optional[str] = None # base64 encoded image


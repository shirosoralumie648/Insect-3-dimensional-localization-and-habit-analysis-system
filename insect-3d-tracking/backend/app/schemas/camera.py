from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class CameraConfigBase(BaseModel):
    """相机配置基础模型"""
    camera_index: int
    name: Optional[str] = None
    width: int
    height: int
    fps: int
    exposure: Optional[float] = None
    gain: Optional[float] = None
    camera_matrix: Optional[Dict[str, Any]] = None
    distortion_coeffs: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, float]] = None  # {x, y, z}
    rotation: Optional[Dict[str, float]] = None  # 四元数 {w, x, y, z}


class CameraConfigCreate(CameraConfigBase):
    """相机配置创建模型"""
    project_id: int


class CameraConfigUpdate(BaseModel):
    """相机配置更新模型"""
    name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    exposure: Optional[float] = None
    gain: Optional[float] = None
    camera_matrix: Optional[Dict[str, Any]] = None
    distortion_coeffs: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, float]] = None
    rotation: Optional[Dict[str, float]] = None


class CameraConfigInDB(CameraConfigBase):
    """数据库中的相机配置模型"""
    id: int
    project_id: int

    class Config:
        orm_mode = True


class CameraConfig(CameraConfigInDB):
    """相机配置响应模型"""
    pass


class CameraConfigList(BaseModel):
    """相机配置列表响应模型"""
    total: int
    items: List[CameraConfig]


class CameraInfo(BaseModel):
    """相机信息模型"""
    camera_index: int
    name: str
    is_available: bool
    capabilities: Dict[str, Any]


class CameraInfoList(BaseModel):
    """可用相机列表响应模型"""
    total: int
    items: List[CameraInfo]


class StreamFrame(BaseModel):
    """视频帧数据模型（用于WebSocket）"""
    camera_index: int
    frame_number: int
    timestamp: float
    image_data: str  # Base64编码的图像数据
    width: int
    height: int
    detection_results: Optional[Dict[str, Any]] = None

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


class VideoBase(BaseModel):
    """视频基础模型"""
    name: str
    camera_index: Optional[int] = None
    duration: Optional[float] = None  # 视频时长(秒)
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    format: Optional[str] = None


class VideoCreate(VideoBase):
    """视频创建模型"""
    project_id: int
    path: str


class VideoUpdate(BaseModel):
    """视频更新模型"""
    name: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    format: Optional[str] = None


class VideoInDB(VideoBase):
    """数据库中的视频模型"""
    id: int
    project_id: int
    path: str
    created_at: datetime

    class Config:
        orm_mode = True


class Video(VideoInDB):
    """视频响应模型"""
    pass


class VideoList(BaseModel):
    """视频列表响应模型"""
    total: int
    items: List[Video]


class RecordingSettings(BaseModel):
    """录制设置"""
    camera_indices: List[int]  # 要录制的相机索引列表
    width: int
    height: int
    fps: int
    format: str = "mp4"  # 视频格式
    codec: str = "h264"  # 编码器
    quality: int = 23  # 编码质量(较低的值表示更高的质量)
    segment_time: Optional[int] = None  # 分段时长(秒)，None表示不分段


class RecordingState(BaseModel):
    """录制状态"""
    camera_index: int
    is_recording: bool
    recording_path: Optional[str] = None
    start_time: Optional[datetime] = None
    elapsed_time: Optional[float] = None
    frame_count: Optional[int] = None
    error: Optional[str] = None


class RecordingStateList(BaseModel):
    """录制状态列表响应模型"""
    total: int
    items: List[RecordingState]


class VideoConversionSettings(BaseModel):
    """视频转换设置"""
    input_path: str
    output_path: Optional[str] = None
    output_format: str = "mp4"
    resize: Optional[Dict[str, int]] = None  # {width, height}
    fps: Optional[int] = None
    start_time: Optional[Union[float, str]] = None
    end_time: Optional[Union[float, str]] = None
    bitrate: Optional[str] = None

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


# Video Schemas
class VideoBase(BaseModel):
    """视频基础模型"""
    name: str
    path: str
    camera_index: Optional[int] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    file_size: Optional[int] = None
    codec_name: Optional[str] = None


class VideoCreate(VideoBase):
    """视频创建模型"""
    project_id: int


class VideoUpdate(BaseModel):
    """视频更新模型"""
    name: Optional[str] = None


class VideoInDB(VideoBase):
    """数据库中的视频模型"""
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class VideoResponse(VideoInDB):
    """视频响应模型"""
    pass


# Recording Settings Schemas
class RecordingSettingsBase(BaseModel):
    """录制设置基础模型"""
    name: str = "Default Recording Settings"
    project_id: int
    output_dir: Optional[str] = None
    filename_prefix: Optional[str] = "video"
    fourcc: Optional[str] = "mp4v"
    fps: Optional[int] = 30
    width: Optional[int] = 1920
    height: Optional[int] = 1080


class RecordingSettingsCreate(RecordingSettingsBase):
    """录制设置创建模型"""
    pass


class RecordingSettingsUpdate(BaseModel):
    """录制设置更新模型"""
    name: Optional[str] = None
    output_dir: Optional[str] = None
    filename_prefix: Optional[str] = None
    fourcc: Optional[str] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class RecordingSettingsInDB(RecordingSettingsBase):
    """数据库中的录制设置模型"""
    id: int

    class Config:
        from_attributes = True


class RecordingSettingsResponse(RecordingSettingsInDB):
    """录制设置响应模型"""
    pass


# Recording Status & Control
class RecordingStatus(BaseModel):
    """录制状态"""
    is_recording: bool
    start_time: Optional[datetime] = None
    elapsed_time: Optional[float] = None
    frame_count: Optional[int] = None
    output_path: Optional[str] = None
    error: Optional[str] = None


# Video Conversion Schemas
class VideoConversionRequest(BaseModel):
    """视频转换请求"""
    input_path: str
    output_path: str
    output_format: str = "mp4"
    video_codec: Optional[str] = "libx264"
    audio_codec: Optional[str] = "aac"
    bitrate: Optional[str] = None
    frame_rate: Optional[int] = None
    resolution: Optional[str] = None # e.g., "1280x720"

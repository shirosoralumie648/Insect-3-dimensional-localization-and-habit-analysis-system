from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AnalysisSettings(BaseModel):
    """行为分析设置"""
    activity_time_interval: int = 60
    heatmap_grid_size: int = 50
    heatmap_projection_plane: str = 'xy'
    behavior_speed_thresholds: Dict[str, float] = Field(default_factory=lambda: {"walking": 0.01, "running": 0.1})
    behavior_turn_threshold: float = 45.0


class AnalysisRequest(BaseModel):
    """分析请求体"""
    session_id: int
    settings: AnalysisSettings


class TrajectoryStats(BaseModel):
    """轨迹统计数据模型"""
    total_distance: float
    average_speed: float
    max_speed: float
    total_duration: float
    active_duration: float


class BehaviorSummary(BaseModel):
    """行为总结模型"""
    behavior: str
    duration: float
    count: int
    average_duration: float


class SpatialHeatmap(BaseModel):
    """空间热图数据模型"""
    grid: List[List[float]]
    x_edges: List[float]
    y_edges: List[float]


class ActivityTimeline(BaseModel):
    """活动时间线数据模型"""
    timestamps: List[datetime]
    speeds: List[float]


class AnalysisResultBase(BaseModel):
    """分析结果基础模型"""
    settings: Dict[str, Any]
    trajectory_stats: Dict[str, Any]
    activity_timeline: Dict[str, Any]
    spatial_heatmap: Dict[str, Any]
    behavior_summary: List[Dict[str, Any]]


class AnalysisResultCreate(AnalysisResultBase):
    """分析结果创建模型"""
    session_id: int
    created_by: int


class AnalysisResultUpdate(BaseModel):
    """分析结果更新模型"""
    settings: Optional[Dict[str, Any]] = None


class AnalysisResultInDB(AnalysisResultBase):
    """数据库中的分析结果模型"""
    id: int
    session_id: int
    created_at: datetime
    created_by: int

    class Config:
        orm_mode = True


class AnalysisResultResponse(AnalysisResultInDB):
    """分析结果响应模型"""
    pass

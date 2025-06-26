from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class BehaviorType(BaseModel):
    """行为类型模型"""
    id: int
    name: str
    description: Optional[str] = None
    color: Optional[str] = None  # 用于可视化的颜色代码


class BehaviorTypeList(BaseModel):
    """行为类型列表响应模型"""
    total: int
    items: List[BehaviorType]


class BehaviorAnalysisSettings(BaseModel):
    """行为分析设置"""
    window_size: int = 30  # 行为分析窗口大小(帧)
    min_trajectory_length: int = 10  # 最小有效轨迹长度
    speed_threshold: Dict[str, float] = Field(default_factory=lambda: {"low": 0.05, "high": 0.5})
    direction_threshold: float = 45.0  # 方向变化阈值(度)
    clustering_algorithm: str = "dbscan"  # 聚类算法


class AnalysisResult(BaseModel):
    """分析结果基础模型"""
    session_id: int
    object_id: int
    behavior_statistics: Dict[str, Any]  # 行为统计信息
    movement_patterns: Dict[str, Any]  # 运动模式
    activity_levels: Dict[str, Any]  # 活动水平
    spatial_preferences: Dict[str, Any]  # 空间偏好
    temporal_patterns: Optional[Dict[str, Any]] = None  # 时序模式


class AnalysisResultList(BaseModel):
    """分析结果列表响应模型"""
    total: int
    items: List[AnalysisResult]


class SpatialHeatmap(BaseModel):
    """空间热图数据模型"""
    session_id: int
    resolution: Dict[str, int]  # {x, y, z} 分辨率
    data: List[Dict[str, Any]]  # 热图数据
    min_value: float
    max_value: float
    colormap: str = "jet"  # 颜色映射


class ActivityTimeline(BaseModel):
    """活动时间线数据模型"""
    session_id: int
    object_id: Optional[int] = None  # 如果为None则表示所有物体
    timeline: List[Dict[str, Any]]  # 时间线数据
    behaviors: Dict[str, Any]  # 行为标签信息


class TrajectoryStatistics(BaseModel):
    """轨迹统计数据模型"""
    session_id: int
    object_id: int
    total_distance: float  # 总移动距离(米)
    average_speed: float  # 平均速度(米/秒)
    max_speed: float  # 最大速度(米/秒)
    movement_directions: Dict[str, float]  # 各方向移动占比
    active_time: float  # 活动时间(秒)
    idle_time: float  # 静止时间(秒)
    bounding_box: Dict[str, Any]  # 活动边界框
    ground_contacts: int  # 与地面接触次数(如果适用)

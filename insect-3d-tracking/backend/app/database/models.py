from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Float, DateTime, Text, JSON, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import datetime
import json

from .session import Base


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    projects = relationship("Project", back_populates="user")





class Project(Base):
    """项目表"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    settings = Column(Text)  # 存储JSON格式的项目设置
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="projects")

    # 关系
    camera_configs = relationship("CameraConfig", back_populates="project", cascade="all, delete-orphan")
    apriltag_configs = relationship("ApriltagConfig", back_populates="project", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="project", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    detection_sessions = relationship("DetectionSession", back_populates="project", cascade="all, delete-orphan")
    recording_settings = relationship("RecordingSettings", back_populates="project", cascade="all, delete-orphan")

    


class CameraConfig(Base):
    """相机配置表"""
    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    camera_index = Column(Integer, nullable=False)  # 摄像头索引（0或1）
    name = Column(String)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    fps = Column(Integer, nullable=False)
    exposure = Column(Float)
    gain = Column(Float)
    camera_matrix = Column(Text)  # 存储JSON格式的相机内参矩阵
    distortion_coeffs = Column(Text)  # 存储JSON格式的畸变系数
    position = Column(Text)  # JSON格式的相机位置(x,y,z)
    rotation = Column(Text)  # JSON格式的相机旋转(四元数)

    # 关系
    project = relationship("Project", back_populates="camera_configs")

    def __init__(self, project_id, camera_index, width, height, fps, **kwargs):
        self.project_id = project_id
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        for key, value in kwargs.items():
            if key in ['camera_matrix', 'distortion_coeffs', 'position', 'rotation'] and value:
                setattr(self, key, json.dumps(value))
            else:
                setattr(self, key, value)
                

class ApriltagConfig(Base):
    """Apriltag配置表"""
    __tablename__ = "apriltag_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    tag_family = Column(String, nullable=False)  # 如"tag36h11"
    tag_id = Column(Integer, nullable=False)
    tag_size = Column(Float, nullable=False)  # 实际大小(米)
    detection_config = Column(Text)  # JSON格式的检测配置

    # 关系
    project = relationship("Project", back_populates="apriltag_configs")

    def __init__(self, project_id, tag_family, tag_id, tag_size, detection_config=None):
        self.project_id = project_id
        self.tag_family = tag_family
        self.tag_id = tag_id
        self.tag_size = tag_size
        if detection_config:
            self.detection_config = json.dumps(detection_config)


class DetectionSession(Base):
    """检测会话表"""
    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    name = Column(String)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime)
    status = Column(String)  # "running", "completed", "error"
    config = Column(Text)  # JSON格式的会话配置

    # 关系
    project = relationship("Project", back_populates="detection_sessions")
    detection_results = relationship("DetectionResult", back_populates="session", cascade="all, delete-orphan")
    tracking_results = relationship("TrackingResult", back_populates="session", cascade="all, delete-orphan")
    trajectories = relationship("Trajectory", back_populates="session", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResult", back_populates="session", cascade="all, delete-orphan")

    def __init__(self, project_id, name=None, config=None):
        self.project_id = project_id
        self.name = name or f"会话_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = "running"
        if config:
            self.config = json.dumps(config)


class DetectionResult(Base):
    """检测结果表"""
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"))
    camera_index = Column(Integer, nullable=False)  # 对应相机索引
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # 帧时间戳
    object_id = Column(Integer)  # 识别的物体ID
    class_id = Column(Integer)  # 类别ID
    class_name = Column(String)  # 类别名称
    confidence = Column(Float)  # 置信度
    x_min = Column(Float)  # 边界框左上角x坐标
    y_min = Column(Float)  # 边界框左上角y坐标
    x_max = Column(Float)  # 边界框右下角x坐标
    y_max = Column(Float)  # 边界框右下角y坐标
    center_x = Column(Float)  # 物体中心x坐标(像素)
    center_y = Column(Float)  # 物体中心y坐标(像素)

    # 关系
    session = relationship("DetectionSession", back_populates="detection_results")

    def __init__(self, session_id, camera_index, frame_number, timestamp, **kwargs):
        self.session_id = session_id
        self.camera_index = camera_index
        self.frame_number = frame_number
        self.timestamp = timestamp
        for key, value in kwargs.items():
            setattr(self, key, value)


class TrackingResult(Base):
    """3D跟踪结果表"""
    __tablename__ = "tracking_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"))
    timestamp = Column(DateTime, default=func.now())
    
    position_3d = Column(JSON)
    confidence = Column(Float)
    class_id = Column(Integer)
    class_name = Column(String)
    
    detection_id1 = Column(Integer, ForeignKey("detection_results.id"), nullable=True)
    detection_id2 = Column(Integer, ForeignKey("detection_results.id"), nullable=True)
    
    tracking_data = Column(JSON)

    # 关系
    session = relationship("DetectionSession", back_populates="tracking_results")
    detection1 = relationship("DetectionResult", foreign_keys=[detection_id1])
    detection2 = relationship("DetectionResult", foreign_keys=[detection_id2])


class Trajectory(Base):
    """轨迹表"""
    __tablename__ = "trajectories"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"))
    object_id = Column(Integer, nullable=False)  # 物体ID
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # 时间戳
    x = Column(Float, nullable=False)  # 3D空间中的x坐标(米)
    y = Column(Float, nullable=False)  # 3D空间中的y坐标(米)
    z = Column(Float, nullable=False)  # 3D空间中的z坐标(米)
    vx = Column(Float)  # x方向速度(米/秒)
    vy = Column(Float)  # y方向速度(米/秒)
    vz = Column(Float)  # z方向速度(米/秒)
    behavior_label = Column(String)  # 行为标签

    # 关系
    session = relationship("DetectionSession", back_populates="trajectories")

    def __init__(self, session_id, object_id, frame_number, timestamp, x, y, z, **kwargs):
        self.session_id = session_id
        self.object_id = object_id
        self.frame_number = frame_number
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        for key, value in kwargs.items():
            setattr(self, key, value)


class AnalysisResult(Base):
    """行为分析结果表"""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id", ondelete="CASCADE"))
    created_at = Column(DateTime, default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))

    settings = Column(Text)  # JSON
    trajectory_stats = Column(Text)  # JSON
    activity_timeline = Column(Text)  # JSON
    spatial_heatmap = Column(Text)  # JSON
    behavior_summary = Column(Text)  # JSON

    # 关系
    session = relationship("DetectionSession", back_populates="analysis_results")
    creator = relationship("User")

    def __init__(self, session_id, created_by, settings=None, trajectory_stats=None, activity_timeline=None, spatial_heatmap=None, behavior_summary=None, **kwargs):
        self.session_id = session_id
        self.created_by = created_by
        if settings:
            self.settings = json.dumps(settings)
        if trajectory_stats:
            self.trajectory_stats = json.dumps(trajectory_stats, default=str)
        if activity_timeline:
            self.activity_timeline = json.dumps(activity_timeline, default=str)
        if spatial_heatmap:
            self.spatial_heatmap = json.dumps(spatial_heatmap, default=str)
        if behavior_summary:
            self.behavior_summary = json.dumps(behavior_summary, default=str)


class Video(Base):
    """视频表"""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    camera_index = Column(Integer)  # 对应相机索引
    created_at = Column(DateTime, default=func.now())
    duration = Column(Float)  # 视频时长(秒)
    width = Column(Integer)  # 视频宽度
    height = Column(Integer)  # 视频高度
    fps = Column(Float)  # 帧率
    format = Column(String)  # 视频格式

    # 关系
    project = relationship("Project", back_populates="videos")

    def __init__(self, project_id, name, path, **kwargs):
        self.project_id = project_id
        self.name = name
        self.path = path
        for key, value in kwargs.items():
            setattr(self, key, value)


class Dataset(Base):
    """数据集表"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    type = Column(String)  # "detection", "behavior", etc.
    image_count = Column(Integer)
    annotation_count = Column(Integer)
    train_split = Column(Float)  # 训练集比例
    val_split = Column(Float)  # 验证集比例
    test_split = Column(Float)  # 测试集比例

    # 关系
    project = relationship("Project", back_populates="datasets")
    models = relationship("Model", back_populates="dataset", cascade="all, delete-orphan")

    def __init__(self, project_id, name, path, type=None, **kwargs):
        self.project_id = project_id
        self.name = name
        self.path = path
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class Model(Base):
    """模型表"""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    type = Column(String)  # "yolov8n", "yolov8s", etc.
    status = Column(String)  # "training", "completed", "error"
    metrics = Column(Text)  # JSON格式的评估指标
    config = Column(Text)  # JSON格式的训练配置

    # 关系
    dataset = relationship("Dataset", back_populates="models")

    def __init__(self, dataset_id, name, path, type=None, **kwargs):
        self.dataset_id = dataset_id
        self.name = name
        self.path = path
        self.type = type
        self.status = "training"
        for key, value in kwargs.items():
            if key in ['metrics', 'config'] and value:
                setattr(self, key, json.dumps(value))
            else:
                setattr(self, key, value)





class RecordingSettings(Base):
    """录制设置表"""
    __tablename__ = "recording_settings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, default="Default Recording Settings")
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    output_dir = Column(String)
    filename_prefix = Column(String, default="video")
    fourcc = Column(String, default="mp4v")
    fps = Column(Integer, default=30)
    width = Column(Integer, default=1920)
    height = Column(Integer, default=1080)

    project = relationship("Project", back_populates="recording_settings")

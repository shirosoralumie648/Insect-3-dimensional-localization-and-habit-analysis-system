from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


class ModelBase(BaseModel):
    """模型基础模型"""
    name: str
    path: str
    type: Optional[str] = None  # "yolov8n", "yolov8s", etc.


class ModelCreate(ModelBase):
    """模型创建模型"""
    dataset_id: int
    config: Optional[Dict[str, Any]] = None


class ModelUpdate(BaseModel):
    """模型更新模型"""
    name: Optional[str] = None
    status: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class ModelInDB(ModelBase):
    """数据库中的模型"""
    id: int
    dataset_id: int
    created_at: datetime
    status: str  # "training", "completed", "error"
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class Model(ModelInDB):
    """模型响应模型"""
    pass


class ModelList(BaseModel):
    """模型列表响应模型"""
    total: int
    items: List[Model]


class TrainingSettings(BaseModel):
    """训练设置"""
    model_type: str  # "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    device: str = "cuda:0"  # "cpu", "cuda:0", etc.
    patience: int = 20  # 早停参数
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    optimizer: str = "SGD"  # "SGD", "Adam", etc.
    data_augmentation: Dict[str, Any] = Field(default_factory=dict)
    pretrained: bool = True
    pretrained_weights: Optional[str] = None


class TrainingProgress(BaseModel):
    """训练进度"""
    model_id: int
    epoch: int
    total_epochs: int
    loss: float
    val_loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mAP50: Optional[float] = None
    mAP50_95: Optional[float] = None
    progress_percentage: float
    time_elapsed: float  # 已用时间(秒)
    estimated_time_remaining: Optional[float] = None  # 预计剩余时间(秒)


class InferenceSettings(BaseModel):
    """推理设置"""
    model_id: int
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: Optional[List[int]] = None
    max_det: int = 300
    device: str = "cuda:0"


class InferenceResult(BaseModel):
    """推理结果"""
    boxes: List[Dict[str, Union[float, int, str]]]  # 检测框列表
    processing_time: float  # 处理时间(毫秒)
    image_size: Dict[str, int]  # 图像尺寸


class EvaluationMetrics(BaseModel):
    """评估指标"""
    model_id: int
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    f1_score: float
    class_metrics: Dict[str, Dict[str, float]]  # 每类指标


class TrainingJobCreate(BaseModel):
    """训练任务创建模型"""
    dataset_id: int
    base_model: str = Field("yolov8n.pt", description="基础模型名称，例如 yolov8n.pt")
    epochs: int = 100
    batch_size: int = 16
    img_size: int = 640
    device: str = Field("", description="训练设备，例如 'cpu', '0', '0,1,2,3'")


class TrainingJobResponse(BaseModel):
    """训练任务响应模型"""
    job_id: str
    model_id: int
    status: str

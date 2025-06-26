from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class DatasetBase(BaseModel):
    """数据集基础模型"""
    name: str
    path: str
    type: Optional[str] = None  # "detection", "behavior", etc.
    image_count: Optional[int] = None
    annotation_count: Optional[int] = None
    train_split: Optional[float] = 0.8
    val_split: Optional[float] = 0.1
    test_split: Optional[float] = 0.1


class DatasetCreate(DatasetBase):
    """数据集创建模型"""
    project_id: int


class DatasetUpdate(BaseModel):
    """数据集更新模型"""
    name: Optional[str] = None
    type: Optional[str] = None
    image_count: Optional[int] = None
    annotation_count: Optional[int] = None
    train_split: Optional[float] = None
    val_split: Optional[float] = None
    test_split: Optional[float] = None


class DatasetInDB(DatasetBase):
    """数据库中的数据集模型"""
    id: int
    project_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Dataset(DatasetInDB):
    """数据集响应模型"""
    pass


class DatasetList(BaseModel):
    """数据集列表响应模型"""
    total: int
    items: List[Dataset]


class AnnotationBase(BaseModel):
    """标注基础模型"""
    image_path: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    class_name: str


class AnnotationCreate(BaseModel):
    """标注创建模型"""
    image_name: str
    coco_annotation: Dict[str, Any]


class AnnotationUpdate(BaseModel):
    """标注更新模型"""
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None


class Annotation(AnnotationBase):
    """标注响应模型"""
    id: int
    dataset_id: int


class AnnotationList(BaseModel):
    """标注列表响应模型"""
    total: int
    items: List[Annotation]


class DatasetSplit(BaseModel):
    """数据集分割结果"""
    train_count: int
    val_count: int
    test_count: int
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]


class DatasetExportFormat(BaseModel):
    """数据集导出格式"""
    format_name: str  # "yolov8", "coco", "voc", etc.
    description: Optional[str] = None


class DatasetExportRequest(BaseModel):
    """数据集导出请求"""
    dataset_id: int
    format: str
    export_path: Optional[str] = None
    include_images: bool = True


class DataAugmentationSettings(BaseModel):
    """数据增强设置"""
    flip: bool = True
    rotate: bool = True
    scale: Dict[str, float] = Field(default_factory=lambda: {"min": 0.8, "max": 1.2})
    translate: Dict[str, float] = Field(default_factory=lambda: {"x": 0.1, "y": 0.1})
    brightness: Dict[str, float] = Field(default_factory=lambda: {"min": -0.1, "max": 0.1})
    contrast: Dict[str, float] = Field(default_factory=lambda: {"min": 0.8, "max": 1.2})
    blur: bool = False
    noise: bool = False

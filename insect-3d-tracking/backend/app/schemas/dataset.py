from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime

# --- Dataset Schemas ---

class DatasetBase(BaseModel):
    """数据集基础模型"""
    name: str
    project_id: int
    description: Optional[str] = None
    type: Optional[str] = "detection"  # e.g., "detection", "behavior"

class DatasetCreate(DatasetBase):
    """数据集创建模型"""
    pass

class DatasetUpdate(BaseModel):
    """数据集更新模型"""
    name: Optional[str] = None
    description: Optional[str] = None

class DatasetInDB(DatasetBase):
    """数据库中的数据集模型"""
    id: int
    path: str
    image_count: int = 0
    annotation_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DatasetResponse(DatasetInDB):
    """数据集响应模型"""
    pass


# --- Annotation Schemas ---

class AnnotationBase(BaseModel):
    """标注基础模型"""
    image_name: str
    # Using a flexible Dict for COCO format
    coco_annotation: Dict[str, Any]

class AnnotationCreate(AnnotationBase):
    """标注创建模型"""
    pass

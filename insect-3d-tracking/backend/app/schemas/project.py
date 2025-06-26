from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ProjectBase(BaseModel):
    """项目基础模型"""
    name: str
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ProjectCreate(ProjectBase):
    """项目创建模型"""
    pass


class ProjectUpdate(BaseModel):
    """项目更新模型"""
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ProjectInDB(ProjectBase):
    """数据库中的项目模型"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class Project(ProjectInDB):
    """项目响应模型"""
    pass


class ProjectList(BaseModel):
    """项目列表响应模型"""
    total: int
    items: List[Project]

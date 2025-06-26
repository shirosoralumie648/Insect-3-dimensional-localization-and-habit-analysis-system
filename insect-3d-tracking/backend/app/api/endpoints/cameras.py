from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Any, List, Optional, Dict

from ...database.session import get_db
from ...database.models import CameraConfig, User, Project
from ...schemas.camera import (
    CameraConfig as CameraConfigSchema,
    CameraConfigCreate,
    CameraConfigUpdate,
    CameraConfigList,
    CameraInfo
)
from ...core.camera import get_available_cameras, get_camera_parameters
from ..deps import get_current_active_user

router = APIRouter()


@router.get("/available", response_model=List[CameraInfo])
def get_available_camera_devices(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取系统可用的相机设备列表
    """
    cameras = get_available_cameras()
    return cameras


@router.get("/", response_model=CameraConfigList)
def read_camera_configs(
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取相机配置列表
    """
    query = db.query(CameraConfig)
    
    if project_id:
        # 确认项目存在且用户有访问权限
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="项目不存在"
            )
        
        if project.user_id != current_user.id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="没有权限访问此项目"
            )
        
        query = query.filter(CameraConfig.project_id == project_id)
    else:
        # 如果没有指定项目，只返回用户有权限访问的项目的相机配置
        project_ids = [
            p.id for p in db.query(Project).filter(
                Project.user_id == current_user.id
            ).all()
        ]
        query = query.filter(CameraConfig.project_id.in_(project_ids))
    
    total = query.count()
    configs = query.offset(skip).limit(limit).all()
    
    return {"total": total, "items": configs}


@router.post("/", response_model=CameraConfigSchema)
def create_camera_config(
    *,
    db: Session = Depends(get_db),
    config_in: CameraConfigCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    创建新的相机配置
    """
    project = db.query(Project).filter(Project.id == config_in.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="没有权限访问此项目")

    db_config = CameraConfig(**config_in.model_dump())
    db.add(db_config)
    db.commit()
    db.refresh(db_config)
    return db_config


@router.get("/{config_id}", response_model=CameraConfigSchema)
def read_camera_config(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取相机配置详情
    """
    config = db.query(CameraConfig).filter(CameraConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="相机配置不存在"
        )
    
    # 确认用户有权限访问该项目的相机配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此项目的相机配置"
        )
    
    return config


@router.put("/{config_id}", response_model=CameraConfigSchema)
def update_camera_config(
    *,
    db: Session = Depends(get_db),
    config_id: int,
    config_in: CameraConfigUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    更新相机配置
    """
    config = db.query(CameraConfig).filter(CameraConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="相机配置不存在"
        )
    
    # 确认用户有权限修改该项目的相机配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限修改此项目的相机配置"
        )
    
    update_data = config_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)
    
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.delete("/{config_id}", response_model=Dict[str, str])
def delete_camera_config(
    *,
    db: Session = Depends(get_db),
    config_id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除相机配置
    """
    config = db.query(CameraConfig).filter(CameraConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="相机配置不存在"
        )

    # 确认用户有权限删除该项目的相机配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )

    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此项目的相机配置"
        )

    db.delete(config)
    db.commit()
    return {"message": "Camera config deleted successfully"}

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Any, List, Optional

from ...database.session import get_db
from ...database.models import Project, User
from ...schemas.project import (
    Project as ProjectSchema,
    ProjectCreate,
    ProjectUpdate,
    ProjectList
)
from ..deps import get_current_active_user

router = APIRouter()


@router.get("/", response_model=ProjectList)
def read_projects(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    name: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取当前用户的项目列表
    """
    query = db.query(Project).filter(Project.user_id == current_user.id)
    
    if name:
        query = query.filter(Project.name.like(f"%{name}%"))
    
    total = query.count()
    projects = query.offset(skip).limit(limit).all()
    
    return {"total": total, "items": projects}


@router.post("/", response_model=ProjectSchema)
def create_project(
    *,
    db: Session = Depends(get_db),
    project_in: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    创建新项目
    """
    project = Project(
        user_id=current_user.id,
        name=project_in.name,
        description=project_in.description,
        config=project_in.config,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectSchema)
def read_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取项目详情
    """
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
    
    return project


@router.put("/{project_id}", response_model=ProjectSchema)
def update_project(
    *,
    db: Session = Depends(get_db),
    project_id: int,
    project_in: ProjectUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    更新项目信息
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限修改此项目"
        )
    
    update_data = project_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", response_model=ProjectSchema)
def delete_project(
    *,
    db: Session = Depends(get_db),
    project_id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除项目
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此项目"
        )
    
    db.delete(project)
    db.commit()
    return project

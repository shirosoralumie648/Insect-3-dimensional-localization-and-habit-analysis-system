from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
from datetime import datetime

from ...database.session import get_db
from ...database.models import (
    User,
    Project,
    DetectionSession,
    TrackingResult,
    AnalysisResult
)
from ...schemas.analysis import (
    AnalysisSettings,
    AnalysisResultCreate,
    AnalysisResultUpdate,
    AnalysisResultResponse,
    BehaviorSummary,
    SpatialHeatmap,
    ActivityTimeline,
    TrajectoryStats
)
from ...core.analysis import TrajectoryAnalyzer
from ..deps import get_current_active_user

router = APIRouter()


@router.post("/sessions/{session_id}/analyze", response_model=AnalysisResultResponse)
def analyze_trajectory(
    session_id: int,
    settings: AnalysisSettings,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    对指定的检测会话进行轨迹和习性分析
    """
    # 检查检测会话是否存在
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="检测会话不存在"
        )
    
    # 检查用户是否有权限访问该项目
    project = db.query(Project).filter(Project.id == session.project_id).first()
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
    
    # 获取该会话的所有3D定位结果
    tracking_results = db.query(TrackingResult).filter(
        TrackingResult.session_id == session_id
    ).order_by(TrackingResult.timestamp).all()
    
    if not tracking_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="该会话没有可分析的3D定位数据"
        )
    
    # 将数据库模型转换为字典
    tracking_data = [
        {
            "timestamp": r.timestamp,
            "position_3d": r.position_3d,
            "class_id": r.class_id,
            "class_name": r.class_name,
            "confidence": r.confidence
        }
        for r in tracking_results
    ]
    
    # 创建轨迹分析器
    analyzer = TrajectoryAnalyzer(tracking_data)
    
    # 执行分析
    basic_stats = analyzer.get_basic_stats()
    activity_timeline = analyzer.get_activity_timeline(
        time_interval_seconds=settings.activity_time_interval
    )
    spatial_heatmap = analyzer.get_spatial_heatmap(
        grid_size=settings.heatmap_grid_size,
        projection_plane=settings.heatmap_projection_plane
    )
    behavior_df = analyzer.classify_behavior(
        speed_thresholds=settings.behavior_speed_thresholds,
        turn_threshold=settings.behavior_turn_threshold
    )
    behavior_summary = analyzer.get_behavior_summary()
    
    # 将分析结果保存到数据库
    analysis_result = AnalysisResult(
        session_id=session_id,
        settings=settings.dict(),
        trajectory_stats=basic_stats,
        activity_timeline=activity_timeline,
        spatial_heatmap=spatial_heatmap,
        behavior_summary=behavior_summary,
        created_by=current_user.id
    )
    
    db.add(analysis_result)
    db.commit()
    db.refresh(analysis_result)
    
    return analysis_result


@router.get("/sessions/{session_id}/results", response_model=List[AnalysisResultResponse])
def get_analysis_results(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取指定检测会话的所有分析结果
    """
    # 检查检测会话是否存在
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="检测会话不存在"
        )
    
    # 检查用户是否有权限访问该项目
    project = db.query(Project).filter(Project.id == session.project_id).first()
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
    
    results = db.query(AnalysisResult).filter(
        AnalysisResult.session_id == session_id
    ).all()
    
    return results


@router.get("/results/{result_id}", response_model=AnalysisResultResponse)
def get_analysis_result(
    result_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取单个分析结果的详情
    """
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查用户是否有权限访问该项目
    session = db.query(DetectionSession).filter(DetectionSession.id == result.session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="检测会话不存在"
        )
    
    project = db.query(Project).filter(Project.id == session.project_id).first()
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
    
    return result


@router.delete("/results/{result_id}", response_model=AnalysisResultResponse)
def delete_analysis_result(
    result_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除分析结果
    """
    result = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="分析结果不存在"
        )
    
    # 检查用户是否有权限访问该项目
    session = db.query(DetectionSession).filter(DetectionSession.id == result.session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="检测会话不存在"
        )
    
    project = db.query(Project).filter(Project.id == session.project_id).first()
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
    
    db.delete(result)
    db.commit()
    
    return result

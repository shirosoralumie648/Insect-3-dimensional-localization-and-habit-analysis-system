from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, WebSocket
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
import os
import uuid
from pathlib import Path
import asyncio

from ...database.session import get_db
from ...database.models import (
    User,
    Project,
    Video,
    RecordingSettings as RecordingSettingsModel
)
from ...schemas.recording import (
    VideoCreate,
    VideoUpdate,
    VideoResponse,
    RecordingSettingsCreate,
    RecordingSettingsUpdate,
    RecordingSettingsResponse,
    RecordingStatus,
    VideoConversionRequest
)
from ...core.recording import VideoRecorder, VideoConverter
from ...core.camera import get_camera_instance
from ..deps import get_current_active_user
from ...config import settings

router = APIRouter()

# 用于存储每个相机录制器实例的字典
recorders: Dict[int, VideoRecorder] = {}


def get_recorder_instance(camera_index: int, settings: RecordingSettingsModel) -> VideoRecorder:
    """获取或创建VideoRecorder实例"""
    if camera_index not in recorders:
        recorders[camera_index] = VideoRecorder(
            camera_index=camera_index,
            output_dir=settings.output_dir or str(settings.VIDEO_DIR),
            filename_prefix=settings.filename_prefix,
            fourcc=settings.fourcc,
            fps=settings.fps,
            frame_size=(settings.width, settings.height) if settings.width and settings.height else None
        )
    return recorders[camera_index]


@router.post("/settings/", response_model=RecordingSettingsResponse)
def create_recording_settings(
    *,
    db: Session = Depends(get_db),
    settings_in: RecordingSettingsCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    创建新的录制设置
    """
    # 检查项目是否存在
    project = db.query(Project).filter(Project.id == settings_in.project_id).first()
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
    
    # 创建录制设置
    db_settings = RecordingSettingsModel(**settings_in.dict())
    db.add(db_settings)
    db.commit()
    db.refresh(db_settings)
    
    return db_settings


@router.get("/settings/{settings_id}", response_model=RecordingSettingsResponse)
def read_recording_settings(
    settings_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取录制设置详情
    """
    db_settings = db.query(RecordingSettingsModel).filter(RecordingSettingsModel.id == settings_id).first()
    if not db_settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="录制设置不存在"
        )
    
    # 检查权限
    project = db.query(Project).filter(Project.id == db_settings.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此录制设置"
        )
    
    return db_settings


@router.post("/start/{camera_index}")
def start_recording(
    camera_index: int,
    settings_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    开始录制
    """
    # 获取录制设置
    recording_settings = db.query(RecordingSettingsModel).filter(RecordingSettingsModel.id == settings_id).first()
    if not recording_settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="录制设置不存在"
        )
    
    # 检查权限
    project = db.query(Project).filter(Project.id == recording_settings.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此录制设置"
        )
    
    # 获取录制器实例
    recorder = get_recorder_instance(camera_index, recording_settings)
    
    try:
        output_path = recorder.start()
        return {
            "message": "录制已开始",
            "output_path": output_path,
            "status": recorder.get_status()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"开始录制失败: {str(e)}"
        )


@router.post("/stop/{camera_index}", response_model=VideoResponse)
def stop_recording(
    camera_index: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    停止录制并保存视频信息到数据库
    """
    if camera_index not in recorders or not recorders[camera_index].is_recording:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该相机未在录制中"
        )
    
    recorder = recorders[camera_index]
    
    try:
        info = recorder.stop()
        
        # 获取项目ID
        # (这里假设录制设置在启动时已验证，但更好的做法是将会话与项目关联)
        # 暂定为从最近的录制设置中获取
        last_settings = db.query(RecordingSettingsModel).order_by(RecordingSettingsModel.id.desc()).first()
        if not last_settings:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法确定视频所属项目"
            )
        
        # 创建视频记录
        video = Video(
            project_id=last_settings.project_id,
            file_path=info["output_path"],
            file_name=os.path.basename(info["output_path"]),
            duration=info["duration"],
            frame_count=info["frame_count"],
            file_size=info["file_size"],
            resolution=f"{recorder.frame_size[0]}x{recorder.frame_size[1]}",
            fps=recorder.fps,
            recorded_at=datetime.fromtimestamp(recorder.start_time) if recorder.start_time else datetime.now(),
            created_by=current_user.id
        )
        
        db.add(video)
        db.commit()
        db.refresh(video)
        
        # 移除录制器实例
        del recorders[camera_index]
        
        return video
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"停止录制失败: {str(e)}"
        )


@router.get("/status/{camera_index}", response_model=RecordingStatus)
def get_recording_status(camera_index: int) -> Any:
    """
    获取指定相机的录制状态
    """
    if camera_index not in recorders:
        return {"is_recording": False}
    
    return recorders[camera_index].get_status()


@router.post("/convert")
def convert_video(
    request: VideoConversionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    转换视频格式 (后台任务)
    """
    if not os.path.exists(request.input_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="输入视频文件不存在"
        )
    
    # 简单的权限检查 (确保文件在项目目录下)
    if not Path(request.input_path).is_relative_to(settings.VIDEO_DIR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此文件"
        )
    
    def conversion_task():
        try:
            VideoConverter.convert(
                input_path=request.input_path,
                output_path=request.output_path,
                output_format=request.output_format,
                video_codec=request.video_codec,
                audio_codec=request.audio_codec,
                bitrate=request.bitrate,
                frame_rate=request.frame_rate,
                resolution=request.resolution
            )
            print(f"视频转换完成: {request.output_path}")
        except Exception as e:
            print(f"视频转换失败: {str(e)}")
    
    background_tasks.add_task(conversion_task)
    
    return {"message": "视频转换任务已开始", "output_path": request.output_path}


@router.get("/videos/", response_model=List[VideoResponse])
def read_videos(
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取视频列表
    """
    query = db.query(Video)
    
    if project_id:
        # 检查权限
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project or (project.user_id != current_user.id and not current_user.is_admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="没有权限访问此项目的视频"
            )
        query = query.filter(Video.project_id == project_id)
    else:
        if not current_user.is_admin:
            project_ids = [p.id for p in db.query(Project).filter(Project.user_id == current_user.id).all()]
            query = query.filter(Video.project_id.in_(project_ids))
    
    videos = query.offset(skip).limit(limit).all()
    return videos


@router.delete("/videos/{video_id}", response_model=VideoResponse)
def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除视频记录和文件
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="视频不存在"
        )
    
    # 检查权限
    project = db.query(Project).filter(Project.id == video.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此视频"
        )
    
    # 删除文件
    try:
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
    except Exception as e:
        # 即使文件删除失败，也继续删除数据库记录
        print(f"删除视频文件失败: {str(e)}")
    
    db.delete(video)
    db.commit()
    
    return video

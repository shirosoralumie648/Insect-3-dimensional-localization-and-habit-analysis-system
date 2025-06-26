import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional, Tuple
import base64
import json
import time
from datetime import datetime
import os
import uuid
from pathlib import Path
import asyncio

from ...database.session import get_db
from ...database.models import (
    ApriltagConfig,
    CameraConfig,
    User,
    Project,
    DetectionSession,
    DetectionResult,
    TrackingResult
)
from ...schemas.detection import (
    DetectionSessionCreate,
    DetectionSessionUpdate,
    DetectionSessionResponse,
    DetectionResultCreate,
    DetectionResultUpdate,
    DetectionResultResponse,
    TrackingResultCreate,
    TrackingResultUpdate,
    TrackingResultResponse,
    DetectionSettings,
    DetectionRequest,
    LocalizationSettings,
    LocalizationResult
)
from ...core.detection import InsectDetector, Localizer3D
from ...core.camera import get_camera_instance
from ..deps import get_current_active_user
from ...config import settings

router = APIRouter()


@router.post("/sessions/", response_model=DetectionSessionResponse)
def create_detection_session(
    *,
    db: Session = Depends(get_db),
    session_in: DetectionSessionCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    创建新的检测会话
    """
    # 检查项目是否存在以及用户是否有访问权限
    project = db.query(Project).filter(Project.id == session_in.project_id).first()
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
    
    # 创建检测会话
    session = DetectionSession(
        project_id=session_in.project_id,
        name=session_in.name,
        config=session_in.config.model_dump() if session_in.config else None
    )
    session.model_path = session_in.model_path
    session.status = "created"
    session.created_by = current_user.id
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


@router.get("/sessions/", response_model=List[DetectionSessionResponse])
def read_detection_sessions(
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取检测会话列表
    """
    query = db.query(DetectionSession)
    
    if project_id:
        # 检查项目是否存在以及用户是否有访问权限
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
        
        query = query.filter(DetectionSession.project_id == project_id)
    else:
        # 如果没有指定项目，则只返回用户有权限访问的项目的会话
        if not current_user.is_admin:
            project_ids = [
                p.id for p in db.query(Project).filter(
                    Project.user_id == current_user.id
                ).all()
            ]
            query = query.filter(DetectionSession.project_id.in_(project_ids))
    
    sessions = query.offset(skip).limit(limit).all()
    return sessions


@router.get("/sessions/{session_id}", response_model=DetectionSessionResponse)
def read_detection_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取检测会话详情
    """
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
    
    return session


@router.put("/sessions/{session_id}", response_model=DetectionSessionResponse)
def update_detection_session(
    *,
    db: Session = Depends(get_db),
    session_id: int,
    session_in: DetectionSessionUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    更新检测会话
    """
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
    
    update_data = session_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session


@router.delete("/sessions/{session_id}", response_model=DetectionSessionResponse)
def delete_detection_session(
    *,
    db: Session = Depends(get_db),
    session_id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除检测会话
    """
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
    
    db.delete(session)
    db.commit()
    
    return session


@router.post("/detect")
async def detect_insects(
    request: DetectionRequest,
    camera_index: Optional[int] = None,
    image: Optional[UploadFile] = None,
    base64_image: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    检测昆虫
    可以从相机、上传文件或Base64编码图像进行检测
    """
    # 获取图像
    if camera_index is not None:
        # 从相机获取图像
        camera = get_camera_instance(camera_index)
        if not camera.is_running:
            success = camera.open()
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="无法打开相机"
                )
        
        ret, frame = camera.get_frame()
        if not ret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取相机图像帧失败"
            )
    
    elif image is not None:
        # 从上传文件获取图像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法解码图像文件"
            )
    
    elif base64_image is not None:
        # 从Base64编码获取图像
        try:
            # 移除可能的"data:image/jpeg;base64,"前缀
            if "base64," in base64_image:
                base64_image = base64_image.split("base64,")[1]
            
            imgdata = base64.b64decode(base64_image)
            nparr = np.frombuffer(imgdata, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("无法解码Base64图像")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无法解码Base64图像: {str(e)}"
            )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="需要提供camera_index、image或base64_image之一"
        )
    
    # 创建检测器
    detector = InsectDetector(
        model_path=request.settings.model_path,
        confidence=request.settings.confidence,
        device=request.settings.device
    )
    
    # 执行检测
    detections = detector.detect(
        frame,
        classes=request.settings.classes
    )
    
    # 如果需要保存检测结果到数据库
    if request.session_id:
        try:
            # 检查会话是否存在
            session = db.query(DetectionSession).filter(
                DetectionSession.id == request.session_id
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="检测会话不存在"
                )
            
            # 保存检测结果
            for det in detections:
                result = DetectionResult(
                    session_id=request.session_id,
                    timestamp=datetime.now(),
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    detection_data=det
                )
                db.add(result)
            
            db.commit()
        except Exception as e:
            db.rollback()
            # 不中断API调用，但记录错误
            print(f"保存检测结果失败: {str(e)}")
    
    # 如果需要，绘制结果
    if request.settings.draw_detections:
        result_frame = detector.draw_detections(frame, detections)
        
        # 转换为JPEG格式的Base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        jpg_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "detections": detections,
            "count": len(detections),
            "image_width": frame.shape[1],
            "image_height": frame.shape[0],
            "result_image": jpg_base64
        }
    
    return {
        "detections": detections,
        "count": len(detections),
        "image_width": frame.shape[1],
        "image_height": frame.shape[0]
    }


@router.post("/localize")
async def localize_3d(
    settings: LocalizationSettings,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    使用双目相机进行3D定位
    """
    # 检查相机配置是否存在
    camera_config1 = db.query(CameraConfig).filter(
        CameraConfig.id == settings.camera_config_id1
    ).first()
    
    camera_config2 = db.query(CameraConfig).filter(
        CameraConfig.id == settings.camera_config_id2
    ).first()
    
    if not camera_config1 or not camera_config2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="相机配置不存在"
        )
    
    # 检查相机是否已标定
    if not camera_config1.is_calibrated or not camera_config2.is_calibrated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="相机未标定，请先进行相机标定"
        )
    
    # 获取相机参数
    camera_matrix1 = np.array(camera_config1.camera_matrix, dtype=np.float32)
    dist_coeffs1 = np.array(camera_config1.dist_coeffs, dtype=np.float32)
    camera_matrix2 = np.array(camera_config2.camera_matrix, dtype=np.float32)
    dist_coeffs2 = np.array(camera_config2.dist_coeffs, dtype=np.float32)
    
    # 相机相对位姿
    R = np.array(settings.rotation_matrix, dtype=np.float32)
    T = np.array(settings.translation_vector, dtype=np.float32)
    
    # 打开相机
    camera1 = get_camera_instance(camera_config1.camera_index)
    if not camera1.is_running:
        success = camera1.open(
            width=camera_config1.width,
            height=camera_config1.height,
            fps=camera_config1.fps
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法打开相机1"
            )
    
    camera2 = get_camera_instance(camera_config2.camera_index)
    if not camera2.is_running:
        success = camera2.open(
            width=camera_config2.width,
            height=camera_config2.height,
            fps=camera_config2.fps
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法打开相机2"
            )
    
    # 获取图像
    ret1, frame1 = camera1.get_frame()
    if not ret1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取相机1图像帧失败"
        )
    
    ret2, frame2 = camera2.get_frame()
    if not ret2:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取相机2图像帧失败"
        )
    
    # 创建昆虫检测器
    detector = InsectDetector(
        model_path=settings.model_path,
        confidence=settings.confidence,
        device=settings.device
    )
    
    # 执行检测
    detections1 = detector.detect(
        frame1,
        classes=settings.classes
    )
    
    detections2 = detector.detect(
        frame2,
        classes=settings.classes
    )
    
    # 创建3D定位器
    localizer = Localizer3D(
        camera_matrix1,
        dist_coeffs1,
        camera_matrix2,
        dist_coeffs2,
        R,
        T
    )
    
    # 执行3D定位
    localization_results = localizer.localize_detections(detections1, detections2)
    
    # 如果需要保存定位结果到数据库
    if settings.session_id:
        try:
            # 检查会话是否存在
            session = db.query(DetectionSession).filter(
                DetectionSession.id == settings.session_id
            ).first()
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="检测会话不存在"
                )
            
            # 保存定位结果
            timestamp = datetime.now()
            for result in localization_results:
                tracking_result = TrackingResult(
                    session_id=settings.session_id,
                    timestamp=timestamp,
                    position_3d=result["position_3d"],
                    detection_id1=None,  # 可以关联到DetectionResult
                    detection_id2=None,  # 可以关联到DetectionResult
                    confidence=result["confidence"],
                    class_id=result["class_id"],
                    class_name=result["class_name"],
                    tracking_data=result
                )
                db.add(tracking_result)
            
            db.commit()
        except Exception as e:
            db.rollback()
            # 不中断API调用，但记录错误
            print(f"保存定位结果失败: {str(e)}")
    
    # 如果需要，绘制结果
    if settings.draw_detections:
        # 绘制第一个相机的检测结果
        result_frame1 = detector.draw_detections(frame1, [r["detection1"] for r in localization_results])
        
        # 绘制第二个相机的检测结果
        result_frame2 = detector.draw_detections(frame2, [r["detection2"] for r in localization_results])
        
        # 转换为JPEG格式的Base64
        _, buffer1 = cv2.imencode('.jpg', result_frame1)
        jpg_base64_1 = base64.b64encode(buffer1).decode('utf-8')
        
        _, buffer2 = cv2.imencode('.jpg', result_frame2)
        jpg_base64_2 = base64.b64encode(buffer2).decode('utf-8')
        
        return {
            "localization_results": localization_results,
            "count": len(localization_results),
            "result_image1": jpg_base64_1,
            "result_image2": jpg_base64_2
        }
    
    return {
        "localization_results": localization_results,
        "count": len(localization_results)
    }

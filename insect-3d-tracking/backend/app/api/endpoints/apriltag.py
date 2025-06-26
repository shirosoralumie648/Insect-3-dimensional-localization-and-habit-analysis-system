import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks
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
from ...database.models import ApriltagConfig, CameraConfig, User, Project
from ...schemas.apriltag import (
    ApriltagConfig as ApriltagConfigSchema,
    ApriltagConfigCreate,
    ApriltagConfigUpdate,
    ApriltagConfigList,
    ApriltagDetection as ApriltagDetectionSchema,
    ApriltagDetectionRequest,
    ApriltagDetectionResult,
    CalibrationSettings
)
from ...core.apriltag import ApriltagDetector, ApriltagDetection, calibrate_camera, estimate_pose
from ...core.camera import get_camera_instance
from ..deps import get_current_active_user
from ...config import settings

router = APIRouter()


@router.get("/configs/", response_model=ApriltagConfigList)
def read_apriltag_configs(
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取Apriltag配置列表
    """
    query = db.query(ApriltagConfig)
    
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
        
        query = query.filter(ApriltagConfig.project_id == project_id)
    else:
        # 如果没有指定项目，只返回用户有权限访问的项目的配置
        project_ids = [
            p.id for p in db.query(Project).filter(
                Project.user_id == current_user.id
            ).all()
        ]
        query = query.filter(ApriltagConfig.project_id.in_(project_ids))
    
    total = query.count()
    configs = query.offset(skip).limit(limit).all()
    
    return {"total": total, "items": configs}


@router.post("/configs/", response_model=ApriltagConfigSchema)
def create_apriltag_config(
    *,
    db: Session = Depends(get_db),
    config_in: ApriltagConfigCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    创建新的Apriltag配置
    """
    # 确认项目存在且用户有访问权限
    project = db.query(Project).filter(Project.id == config_in.project_id).first()
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
    
    # 创建配置
    config = ApriltagConfig(
        project_id=config_in.project_id,
        name=config_in.name,
        family=config_in.family,
        tag_size=config_in.tag_size,
        tag_spacing=config_in.tag_spacing,
        config=config_in.config
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.get("/configs/{config_id}", response_model=ApriltagConfigSchema)
def read_apriltag_config(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    获取Apriltag配置详情
    """
    config = db.query(ApriltagConfig).filter(ApriltagConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apriltag配置不存在"
        )
    
    # 确认用户有权限访问该项目的配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此项目的Apriltag配置"
        )
    
    return config


@router.put("/configs/{config_id}", response_model=ApriltagConfigSchema)
def update_apriltag_config(
    *,
    db: Session = Depends(get_db),
    config_id: int,
    config_in: ApriltagConfigUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    更新Apriltag配置
    """
    config = db.query(ApriltagConfig).filter(ApriltagConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apriltag配置不存在"
        )
    
    # 确认用户有权限修改该项目的配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限修改此项目的Apriltag配置"
        )
    
    update_data = config_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(config, field, value)
    
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


@router.delete("/configs/{config_id}", response_model=ApriltagConfigSchema)
def delete_apriltag_config(
    *,
    db: Session = Depends(get_db),
    config_id: int,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    删除Apriltag配置
    """
    config = db.query(ApriltagConfig).filter(ApriltagConfig.id == config_id).first()
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apriltag配置不存在"
        )
    
    # 确认用户有权限删除该项目的配置
    project = db.query(Project).filter(Project.id == config.project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    if project.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此项目的Apriltag配置"
        )
    
    db.delete(config)
    db.commit()
    return config


@router.post("/detect", response_model=ApriltagDetectionResult)
async def detect_apriltags(
    request: ApriltagDetectionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    检测Apriltag.
    输入为Base64编码的图像.
    """
    settings = request.settings

    # 获取Apriltag配置
    apriltag_config = db.query(ApriltagConfig).filter(ApriltagConfig.id == settings.apriltag_config_id).first()
    if not apriltag_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apriltag配置不存在"
        )

    # 确认用户有权限访问该项目
    project = db.query(Project).filter(Project.id == apriltag_config.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限访问此项目的Apriltag配置"
        )

    # 从Base64编码获取图像
    try:
        image_base64 = request.image_base64
        # 移除可能的"data:image/jpeg;base64,"前缀
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
        
        imgdata = base64.b64decode(image_base64)
        nparr = np.frombuffer(imgdata, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码Base64图像")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无法解码Base64图像: {str(e)}"
        )

    # 创建检测器
    detector = ApriltagDetector(
        family=apriltag_config.tag_family,
        nthreads=settings.nthreads,
        quad_decimate=settings.quad_decimate,
        quad_sigma=settings.quad_sigma,
        refine_edges=settings.refine_edges,
        decode_sharpening=settings.decode_sharpening,
        debug=settings.debug
    )
    
    # 执行检测
    detections = detector.detect(frame)
    
    # 准备结果
    detection_objects = []
    for detection in detections:
        def corners_to_dict(corners):
            return [{'x': float(c[0]), 'y': float(c[1])} for c in corners]

        detection_objects.append(
            ApriltagDetectionSchema(
                tag_family=detection.tag_family.decode('utf-8'),
                tag_id=detection.tag_id,
                center={'x': float(detection.center[0]), 'y': float(detection.center[1])},
                corners=corners_to_dict(detection.corners),
                pose=None # Not implemented yet
            )
        )
    
    return ApriltagDetectionResult(
        detections=detection_objects,
        timestamp=datetime.utcnow()
    )


@router.post("/calibrate")
async def calibrate_camera_endpoint(
    settings: CalibrationSettings,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    使用Apriltag标签进行相机标定
    """
    # 检查相机配置是否存在
    camera_config = db.query(CameraConfig).filter(CameraConfig.id == settings.camera_config_id).first()
    if not camera_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="相机配置不存在"
        )
    
    # 确认用户有权限访问该项目
    project = db.query(Project).filter(Project.id == camera_config.project_id).first()
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
    
    # 获取Apriltag配置
    apriltag_config = db.query(ApriltagConfig).filter(ApriltagConfig.id == settings.apriltag_config_id).first()
    if not apriltag_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Apriltag配置不存在"
        )
    
    # 创建检测器
    detector = ApriltagDetector(
        family=apriltag_config.tag_family,
        nthreads=settings.nthreads,
        quad_decimate=settings.quad_decimate,
        quad_sigma=settings.quad_sigma,
        refine_edges=settings.refine_edges,
        decode_sharpening=settings.decode_sharpening
    )
    
    # 打开相机
    camera = get_camera_instance(camera_config.camera_index)
    if not camera.is_running:
        success = camera.open(
            width=camera_config.width,
            height=camera_config.height,
            fps=camera_config.fps
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法打开相机"
            )
    
    # 收集标定图像
    image_points = []  # 2D点列表(每个标签的每个图像的角点)
    object_points = []  # 3D点列表(每个标签的每个图像的角点在世界坐标中的位置)
    
    for _ in range(settings.num_images):
        # 获取图像
        ret, frame = camera.get_frame()
        if not ret:
            continue
        
        # 检测标签
        detections = detector.detect(frame)
        
        if not detections:
            continue
        
        # 计算单个图像的点
        frame_image_points = []
        frame_object_points = []
        
        tag_size = apriltag_config.tag_size  # 标签尺寸(米)
        half_size = tag_size / 2.0
        
        # 标签的世界坐标系中的3D点(假设标签在z=0平面上)
        tag_object_points = [
            (-half_size, -half_size, 0.0),  # 左上
            (half_size, -half_size, 0.0),   # 右上
            (half_size, half_size, 0.0),    # 右下
            (-half_size, half_size, 0.0),   # 左下
        ]
        
        for detection in detections:
            # 添加检测到的角点作为图像点
            for corner in detection.corners:
                frame_image_points.append(corner)
            
            # 添加对应的3D点
            for point in tag_object_points:
                frame_object_points.append(point)
        
        if frame_image_points and frame_object_points:
            image_points.append(frame_image_points)
            object_points.append(frame_object_points)
            
            # 如果已经收集到足够的图像，可以提前结束
            if len(image_points) >= settings.min_images:
                break
        
        # 短暂延迟以获取不同的视角
        await asyncio.sleep(settings.capture_delay)
    
    # 如果没有收集到足够的图像，则返回错误
    if len(image_points) < settings.min_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"未能收集到足够的标定图像(需要{settings.min_images}张，仅收集到{len(image_points)}张)"
        )
    
    # 执行相机标定
    image_size = (camera_config.width, camera_config.height)
    camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(
        image_points,
        object_points,
        image_size
    )
    
    # 更新相机配置
    update_data = {
        CameraConfig.camera_matrix: json.dumps(camera_matrix.tolist()),
        CameraConfig.distortion_coeffs: json.dumps(dist_coeffs.tolist()),
        CameraConfig.is_calibrated: True
    }
    db.query(CameraConfig).filter(CameraConfig.id == settings.camera_config_id).update(update_data)
    db.commit()

    return {
        "success": True,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coeffs": dist_coeffs.tolist(),
        "reprojection_error": float(reprojection_error),
        "num_images_used": len(image_points)
    }

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
import asyncio
import base64
import json
import time
from datetime import datetime

from ...database.session import get_db
from ...database.models import CameraConfig, User, Project
from ...schemas.camera import (
    CameraSettings,
    CameraProperty,
    CameraStatus
)
from ...core.camera import get_camera_instance, CameraManager
from ..deps import get_current_active_user

router = APIRouter()


@router.get("/status/{camera_index}", response_model=CameraStatus)
def get_camera_status(
    camera_index: int,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取相机状态
    """
    try:
        camera = get_camera_instance(camera_index)
        return {
            "camera_index": camera_index,
            "is_running": camera.is_running,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取相机状态失败: {str(e)}"
        )


@router.post("/open/{camera_index}", response_model=CameraStatus)
def open_camera(
    camera_index: int,
    settings: CameraSettings,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    打开相机
    """
    try:
        camera = get_camera_instance(camera_index)
        success = camera.open(
            width=settings.width,
            height=settings.height,
            fps=settings.fps
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法打开相机"
            )
        
        return {
            "camera_index": camera_index,
            "is_running": camera.is_running,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"打开相机失败: {str(e)}"
        )


@router.post("/close/{camera_index}", response_model=CameraStatus)
def close_camera(
    camera_index: int,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    关闭相机
    """
    try:
        camera = get_camera_instance(camera_index)
        camera.close()
        
        return {
            "camera_index": camera_index,
            "is_running": camera.is_running,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"关闭相机失败: {str(e)}"
        )


@router.post("/property/{camera_index}", response_model=CameraProperty)
def set_camera_property(
    camera_index: int,
    prop: CameraProperty,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    设置相机属性
    """
    try:
        camera = get_camera_instance(camera_index)
        if not camera.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="相机未打开"
            )
        
        # 转换属性ID
        prop_id_map = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "hue": cv2.CAP_PROP_HUE,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "white_balance": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
            "auto_wb": cv2.CAP_PROP_AUTO_WB,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
        }
        
        if prop.property_name not in prop_id_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的属性: {prop.property_name}"
            )
        
        prop_id = prop_id_map[prop.property_name]
        success = camera.set_property(prop_id, prop.value)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"设置属性 {prop.property_name} 失败"
            )
        
        # 获取实际设置的值
        actual_value = camera.get_property(prop_id)
        
        return {
            "property_name": prop.property_name,
            "value": actual_value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设置相机属性失败: {str(e)}"
        )


@router.get("/property/{camera_index}/{property_name}", response_model=CameraProperty)
def get_camera_property(
    camera_index: int,
    property_name: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取相机属性
    """
    try:
        camera = get_camera_instance(camera_index)
        if not camera.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="相机未打开"
            )
        
        # 转换属性ID
        prop_id_map = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "hue": cv2.CAP_PROP_HUE,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "white_balance": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
            "auto_wb": cv2.CAP_PROP_AUTO_WB,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
        }
        
        if property_name not in prop_id_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的属性: {property_name}"
            )
        
        prop_id = prop_id_map[property_name]
        value = camera.get_property(prop_id)
        
        return {
            "property_name": property_name,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取相机属性失败: {str(e)}"
        )


@router.get("/snapshot/{camera_index}")
def get_camera_snapshot(
    camera_index: int,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取相机当前帧的快照
    """
    try:
        camera = get_camera_instance(camera_index)
        if not camera.is_running:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="相机未打开"
            )
        
        ret, frame = camera.get_frame()
        if not ret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="获取图像帧失败"
            )
        
        # 将图像转换为JPEG格式
        _, jpeg_data = cv2.imencode('.jpg', frame)
        
        return StreamingResponse(
            iter([jpeg_data.tobytes()]),
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取相机快照失败: {str(e)}"
        )


async def stream_generator(camera_index: int):
    """
    视频流生成器
    """
    camera = get_camera_instance(camera_index)
    
    if not camera.is_running:
        success = camera.open()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法打开相机"
            )
    
    try:
        while camera.is_running:
            ret, frame = camera.get_frame()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # 将图像转换为JPEG格式
            _, jpeg_data = cv2.imencode('.jpg', frame)
            
            # 创建multipart响应
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg_data.tobytes() + b'\r\n'
            )
            
            # 控制帧率
            await asyncio.sleep(1.0 / camera.fps)
    except Exception as e:
        print(f"Stream error: {str(e)}")
        camera.close()


@router.get("/stream/{camera_index}")
def video_feed(
    camera_index: int,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取相机视频流(MJPEG)
    """
    return StreamingResponse(
        stream_generator(camera_index),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.websocket("/ws/stream/{camera_index}")
async def websocket_stream(websocket: WebSocket, camera_index: int):
    """
    通过WebSocket获取视频流
    """
    await websocket.accept()
    
    camera = get_camera_instance(camera_index)
    if not camera.is_running:
        success = camera.open()
        if not success:
            await websocket.send_text(json.dumps({
                "error": "无法打开相机"
            }))
            await websocket.close()
            return
    
    try:
        while camera.is_running:
            ret, frame = camera.get_frame()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            
            # 降低分辨率以提高传输效率
            scale_factor = 0.5  # 根据需要调整
            if scale_factor != 1.0:
                frame = cv2.resize(
                    frame, 
                    (0, 0), 
                    fx=scale_factor, 
                    fy=scale_factor
                )
            
            # 将图像转换为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Base64编码
            jpg_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构造消息
            message = json.dumps({
                "image": jpg_base64,
                "timestamp": time.time(),
                "camera_index": camera_index,
                "frame_width": frame.shape[1],
                "frame_height": frame.shape[0]
            })
            
            # 发送消息
            await websocket.send_text(message)
            
            # 控制帧率
            await asyncio.sleep(1.0 / camera.fps)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        if websocket.client_state != WebSocket.CLIENT_DISCONNECTED:
            await websocket.close()

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import platform
import json
import os
from pathlib import Path
from ..schemas.camera import CameraInfo


def get_available_cameras() -> List[CameraInfo]:
    """
    获取系统中可用的摄像头设备
    
    Returns:
        List[CameraInfo]: 可用摄像头列表
    """
    cameras = []
    
    # 检测操作系统类型
    system = platform.system()
    
    if system == "Windows":
        # Windows系统下最多尝试10个索引
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cameras.append(
                        CameraInfo(
                            index=i,
                            name=f"Camera {i}",
                            resolution=f"{width}x{height}",
                            fps=fps
                        )
                    )
                cap.release()
    
    elif system == "Linux":
        # Linux系统下检查/dev/video*设备
        import glob
        devices = glob.glob("/dev/video*")
        devices.sort()
        
        for dev in devices:
            idx = int(dev.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cameras.append(
                        CameraInfo(
                            index=idx,
                            name=f"Camera {idx}",
                            resolution=f"{width}x{height}",
                            fps=fps
                        )
                    )
                cap.release()
    
    elif system == "Darwin":  # macOS
        # macOS系统下最多尝试10个索引
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cameras.append(
                        CameraInfo(
                            index=i,
                            name=f"Camera {i}",
                            resolution=f"{width}x{height}",
                            fps=fps
                        )
                    )
                cap.release()
    
    return cameras


def get_camera_parameters(camera_index: int) -> Dict[str, Any]:
    """
    获取相机的参数信息
    
    Args:
        camera_index (int): 相机索引
    
    Returns:
        Dict[str, Any]: 相机参数信息
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"error": "无法打开相机"}
    
    # 获取相机基本参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取支持的分辨率（这只是一些常见的分辨率，实际支持的分辨率可能因设备而异）
    resolutions = [
        "640x480",
        "800x600",
        "1280x720",
        "1920x1080",
        "2560x1440",
        "3840x2160"
    ]
    
    # 获取其他参数
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    # 获取一帧用于测试
    ret, frame = cap.read()
    has_frame = ret
    
    # 释放相机
    cap.release()
    
    return {
        "index": camera_index,
        "width": width,
        "height": height,
        "fps": fps,
        "current_resolution": f"{width}x{height}",
        "supported_resolutions": resolutions,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "hue": hue,
        "gain": gain,
        "exposure": exposure,
        "has_frame": has_frame
    }


_camera_instances = {}


class CameraManager:
    """
    相机管理类，用于控制相机的开启、关闭、参数设置等
    """
    
    def __init__(self, camera_index: int):
        """
        初始化相机管理器
        
        Args:
            camera_index (int): 相机索引
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.width = 640
        self.height = 480
        self.fps = 30
    
    def open(self, width: int = None, height: int = None, fps: int = None) -> bool:
        """
        打开相机
        
        Args:
            width (int, optional): 宽度. Defaults to None.
            height (int, optional): 高度. Defaults to None.
            fps (int, optional): 帧率. Defaults to None.
        
        Returns:
            bool: 是否成功打开
        """
        if self.is_running:
            return True
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            return False
        
        # 设置分辨率
        if width is not None and height is not None:
            self.width = width
            self.height = height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置帧率
        if fps is not None:
            self.fps = fps
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.is_running = True
        return True
    
    def close(self) -> None:
        """
        关闭相机
        """
        if self.cap and self.is_running:
            self.is_running = False
            self.cap.release()
            self.cap = None
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        获取一帧图像
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (是否成功, 图像数据)
        """
        if not self.is_running or not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        return True, frame
    
    def set_property(self, property_id: int, value: float) -> bool:
        """
        设置相机属性
        
        Args:
            property_id (int): 属性ID, 例如cv2.CAP_PROP_BRIGHTNESS
            value (float): 属性值
        
        Returns:
            bool: 是否设置成功
        """
        if not self.is_running or not self.cap:
            return False
        
        return self.cap.set(property_id, value)
    
    def get_property(self, property_id: int) -> float:
        """
        获取相机属性
        
        Args:
            property_id (int): 属性ID, 例如cv2.CAP_PROP_BRIGHTNESS
        
        Returns:
            float: 属性值
        """
        if not self.is_running or not self.cap:
            return 0.0
        
        return self.cap.get(property_id)


def get_camera_instance(camera_index: int) -> CameraManager:
    """
    获取相机实例，如果不存在则创建新的实例
    
    Args:
        camera_index (int): 相机索引
    
    Returns:
        CameraManager: 相机管理器实例
    """
    global _camera_instances
    
    if camera_index not in _camera_instances:
        _camera_instances[camera_index] = CameraManager(camera_index)
    
    return _camera_instances[camera_index]

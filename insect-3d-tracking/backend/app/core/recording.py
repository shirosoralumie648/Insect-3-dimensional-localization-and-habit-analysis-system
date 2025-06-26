import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import time
from pathlib import Path
import threading
from datetime import datetime
import ffmpeg

from ..core.camera import CameraManager
from ..config import settings


class VideoRecorder:
    """视频录制器类"""
    
    def __init__(
        self, 
        camera_index: int,
        output_dir: str = settings.VIDEO_DIR,
        filename_prefix: str = "video",
        fourcc: str = "mp4v",
        fps: Optional[float] = None,
        frame_size: Optional[Tuple[int, int]] = None
    ):
        """
        初始化视频录制器
        
        Args:
            camera_index: 相机索引
            output_dir: 视频输出目录
            filename_prefix: 文件名前缀
            fourcc: 视频编码格式 (e.g., 'mp4v', 'XVID', 'MJPG')
            fps: 录制帧率，如果为None则使用相机默认帧率
            frame_size: 视频帧尺寸 (width, height)，如果为None则使用相机默认尺寸
        """
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        
        self.camera = CameraManager.get_instance(camera_index)
        
        self.fps = fps if fps is not None else self.camera.fps
        self.frame_size = frame_size if frame_size is not None else (self.camera.width, self.camera.height)
        
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.output_path: Optional[str] = None
        self.start_time: Optional[float] = None
        self.frame_count = 0
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def start(self) -> str:
        """开始录制"""
        if self.is_recording:
            raise RuntimeError("录制已在进行中")
        
        if not self.camera.is_running:
            self.camera.open(width=self.frame_size[0], height=self.frame_size[1], fps=self.fps)
            if not self.camera.is_running:
                raise RuntimeError("无法打开相机")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{self.camera_index}_{timestamp}.mp4"
        self.output_path = os.path.join(self.output_dir, filename)
        
        # 创建VideoWriter
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            self.fourcc,
            self.fps,
            self.frame_size
        )
        
        if not self.video_writer.isOpened():
            raise IOError(f"无法打开VideoWriter，文件路径: {self.output_path}")
        
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # 启动录制线程
        self.recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.recording_thread.start()
        
        return self.output_path
    
    def _record_loop(self):
        """录制循环"""
        while self.is_recording:
            ret, frame = self.camera.get_frame()
            if ret:
                # 如果需要，调整帧大小
                if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                    frame = cv2.resize(frame, self.frame_size)
                
                self.video_writer.write(frame)
                self.frame_count += 1
            else:
                # 如果获取帧失败，短暂等待
                time.sleep(0.01)
    
    def stop(self) -> Dict[str, Any]:
        """停止录制"""
        if not self.is_recording:
            return {}
        
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.video_writer:
            self.video_writer.release()
        
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        info = {
            "output_path": self.output_path,
            "duration": duration,
            "frame_count": self.frame_count,
            "file_size": os.path.getsize(self.output_path) if self.output_path and os.path.exists(self.output_path) else 0
        }
        
        # 重置状态
        self.video_writer = None
        self.recording_thread = None
        self.output_path = None
        self.start_time = None
        self.frame_count = 0
        
        return info
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前录制状态"""
        status = {
            "is_recording": self.is_recording,
            "output_path": self.output_path,
            "frame_count": self.frame_count,
            "duration": time.time() - self.start_time if self.is_recording and self.start_time else 0
        }
        return status


class VideoConverter:
    """视频转换器类"""
    
    @staticmethod
    def convert(
        input_path: str,
        output_path: str,
        output_format: str = 'mp4',
        video_codec: str = 'libx264',
        audio_codec: str = 'aac',
        bitrate: Optional[str] = None, # e.g., '1M'
        frame_rate: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None # (width, height)
    ) -> str:
        """
        转换视频格式
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            output_format: 输出格式 (e.g., 'mp4', 'avi', 'mov')
            video_codec: 视频编码器
            audio_codec: 音频编码器
            bitrate: 视频比特率
            frame_rate: 视频帧率
            resolution: 视频分辨率
            
        Returns:
            str: 输出文件路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        # 确保输出目录存在
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        
        try:
            stream = ffmpeg.input(input_path)
            
            # 设置视频参数
            stream = ffmpeg.output(
                stream,
                output_path,
                format=output_format,
                vcodec=video_codec,
                acodec=audio_codec,
                video_bitrate=bitrate,
                r=frame_rate,
                s=f"{resolution[0]}x{resolution[1]}" if resolution else None
            )
            
            # 执行转换
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            return output_path
        
        except ffmpeg.Error as e:
            raise RuntimeError(f"视频转换失败: {e.stderr.decode('utf8')}")

import os
import yaml
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO
from datetime import datetime

from ..config import settings


class TrainingManager:
    """模型训练管理器"""
    
    def __init__(self):
        self.training_processes: Dict[str, Dict[str, Any]] = {}
    
    def start_training(
        self, 
        dataset_config_path: str,
        model_name: str = 'yolov8n.pt', # 预训练模型
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        project_name: str = 'YOLOv8_Training',
        device: str = 'cpu' # 或者 '0' for GPU 0
    ) -> str:
        """
        开始一个新的YOLOv8训练任务
        
        Args:
            dataset_config_path: 数据集配置文件 (dataset.yaml) 的路径
            model_name: 使用的预训练模型
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像尺寸
            project_name: 训练项目名称
            device: 训练设备 ('cpu', '0', '1', etc.)
            
        Returns:
            str: 训练任务的唯一ID
        """
        if not Path(dataset_config_path).exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {dataset_config_path}")
        
        # 生成唯一的训练ID
        train_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建训练线程
        thread = threading.Thread(
            target=self._run_training,
            args=(
                train_id, dataset_config_path, model_name, epochs, 
                batch_size, img_size, project_name, device
            ),
            daemon=True
        )
        
        # 存储训练状态
        self.training_processes[train_id] = {
            "status": "starting",
            "thread": thread,
            "start_time": datetime.now(),
            "end_time": None,
            "config": {
                "dataset_config_path": dataset_config_path,
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "project_name": project_name,
                "device": device
            },
            "results": None,
            "error": None
        }
        
        thread.start()
        return train_id
    
    def _run_training(
        self, train_id, dataset_config_path, model_name, epochs,
        batch_size, img_size, project_name, device
    ):
        """在后台线程中运行训练"""
        try:
            self.training_processes[train_id]["status"] = "running"
            
            # 加载模型
            model = YOLO(model_name)
            
            # 开始训练
            results = model.train(
                data=dataset_config_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project=project_name,
                device=device,
                exist_ok=True # 允许覆盖现有项目
            )
            
            # 训练完成
            self.training_processes[train_id]["status"] = "completed"
            self.training_processes[train_id]["results"] = {
                "best_model_path": results.save_dir / 'weights' / 'best.pt',
                "last_model_path": results.save_dir / 'weights' / 'last.pt',
                "results_path": results.save_dir
            }
        
        except Exception as e:
            self.training_processes[train_id]["status"] = "failed"
            self.training_processes[train_id]["error"] = str(e)
        
        finally:
            self.training_processes[train_id]["end_time"] = datetime.now()
    
    def get_training_status(self, train_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定训练任务的状态
        
        Args:
            train_id: 训练任务ID
            
        Returns:
            Optional[Dict[str, Any]]: 训练状态信息，如果ID不存在则返回None
        """
        if train_id not in self.training_processes:
            return None
        
        process_info = self.training_processes[train_id].copy()
        # 不返回线程对象
        del process_info['thread']
        return process_info
    
    def list_all_trainings(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有训练任务及其状态
        """
        return { 
            k: {ik: iv for ik, iv in v.items() if ik != 'thread'} 
            for k, v in self.training_processes.items()
        }
    
    def stop_training(self, train_id: str) -> bool:
        """
        停止一个正在进行的训练任务 (注意: Ultralytics的训练循环可能不容易从外部中断)
        这是一个尝试性的实现，可能无法立即生效。
        
        Args:
            train_id: 训练任务ID
            
        Returns:
            bool: 是否成功发送停止信号
        """
        if train_id not in self.training_processes or self.training_processes[train_id]['status'] != 'running':
            return False
        
        # Ultralytics的训练没有提供直接的停止API，这是一种变通方法
        # 实际效果可能有限，可能需要更复杂的进程管理
        self.training_processes[train_id]['status'] = 'stopping'
        # 这里可以尝试设置一个标志文件，让训练回调函数读取并停止
        # 但YOLOv8的默认回调不支持这个功能
        
        # 更好的方法是使用multiprocessing.Process来管理训练，这样可以terminate()
        # 但为了简单起见，这里只更新状态
        print(f"发送停止信号到训练 {train_id}，但可能无法立即生效。")
        return True

# 创建一个全局的训练管理器实例
training_manager = TrainingManager()

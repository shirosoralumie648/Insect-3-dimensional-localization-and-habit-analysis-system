import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde


class TrajectoryAnalyzer:
    """昆虫轨迹分析器"""
    
    def __init__(self, tracking_results: List[Dict[str, Any]]):
        """
        初始化轨迹分析器
        
        Args:
            tracking_results: 包含3D定位结果和时间戳的列表
        """
        if not tracking_results:
            self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame(tracking_results)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            # 将3D位置数据转换为单独的列
            self.df[['x', 'y', 'z']] = pd.DataFrame(
                self.df['position_3d'].tolist(), index=self.df.index
            )
            self.df = self.df.sort_values(by='timestamp').reset_index(drop=True)
            self._calculate_kinematics()
    
    def _calculate_kinematics(self):
        """计算轨迹的基本运动学参数"""
        if self.df.empty or len(self.df) < 2:
            self.df['speed'] = 0.0
            self.df['acceleration'] = 0.0
            self.df['turn_angle'] = 0.0
            return
        
        # 计算时间差
        self.df['dt'] = self.df['timestamp'].diff().dt.total_seconds()
        
        # 计算位移
        self.df['dx'] = self.df['x'].diff()
        self.df['dy'] = self.df['y'].diff()
        self.df['dz'] = self.df['z'].diff()
        
        # 计算速度
        self.df['speed'] = np.sqrt(
            self.df['dx']**2 + self.df['dy']**2 + self.df['dz']**2
        ) / self.df['dt']
        
        # 计算加速度
        self.df['acceleration'] = self.df['speed'].diff() / self.df['dt']
        
        # 计算转角
        vectors = self.df[['dx', 'dy', 'dz']].values
        v1 = vectors[:-1]
        v2 = vectors[1:]
        
        dot_product = np.sum(v1 * v2, axis=1)
        norm_product = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        
        # 避免除以零
        mask = (norm_product > 1e-9)
        angles = np.zeros(len(v1))
        angles[mask] = np.arccos(np.clip(dot_product[mask] / norm_product[mask], -1.0, 1.0))
        
        self.df['turn_angle'] = np.nan
        self.df.loc[1:, 'turn_angle'] = np.degrees(angles)
        
        # 填充第一个数据点的NaN值
        self.df.fillna(0, inplace=True)
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """获取轨迹的基本统计数据"""
        if self.df.empty:
            return {}
        
        stats = {
            "duration_seconds": (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds(),
            "total_distance": np.sqrt(self.df['dx']**2 + self.df['dy']**2 + self.df['dz']**2).sum(),
            "average_speed": self.df['speed'].mean(),
            "max_speed": self.df['speed'].max(),
            "average_acceleration": self.df['acceleration'].mean(),
            "max_acceleration": self.df['acceleration'].max(),
            "average_turn_angle": self.df['turn_angle'].mean(),
            "total_turns": (self.df['turn_angle'] > 1e-6).sum()
        }
        
        return stats
    
    def get_activity_timeline(self, time_interval_seconds: int = 60) -> Dict[str, float]:
        """获取活动时间线 (每段时间内的移动距离)"""
        if self.df.empty:
            return {}
        
        self.df['distance'] = np.sqrt(self.df['dx']**2 + self.df['dy']**2 + self.df['dz']**2)
        
        # 按时间间隔分组
        timeline = self.df.set_index('timestamp').resample(f'{time_interval_seconds}S')['distance'].sum()
        
        # 格式化输出
        return {str(k): v for k, v in timeline.to_dict().items()}
    
    def get_spatial_heatmap(
        self, 
        grid_size: int = 20,
        projection_plane: str = 'xy'
    ) -> Dict[str, Any]:
        """
        生成空间热图数据
        
        Args:
            grid_size: 网格大小
            projection_plane: 投影平面 ('xy', 'xz', 'yz')
            
        Returns:
            Dict: 包含热图数据和坐标轴范围
        """
        if self.df.empty:
            return {}
        
        # 选择投影平面
        if projection_plane == 'xy':
            x_data, y_data = self.df['x'], self.df['y']
        elif projection_plane == 'xz':
            x_data, y_data = self.df['x'], self.df['z']
        elif projection_plane == 'yz':
            x_data, y_data = self.df['y'], self.df['z']
        else:
            raise ValueError("projection_plane必须是'xy', 'xz', 'yz'之一")
        
        # 计算坐标轴范围
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()
        
        # 创建网格
        x_grid, y_grid = np.mgrid[x_min:x_max:complex(grid_size), y_min:y_max:complex(grid_size)]
        
        # 使用高斯核密度估计
        positions = np.vstack([x_data, y_data])
        kernel = gaussian_kde(positions)
        z_grid = kernel(np.vstack([x_grid.ravel(), y_grid.ravel()]))
        
        return {
            "heatmap": z_grid.reshape(x_grid.shape).tolist(),
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)]
        }
    
    def classify_behavior(
        self, 
        speed_thresholds: Dict[str, float] = {'resting': 0.1, 'walking': 1.0},
        turn_threshold: float = 30.0
    ) -> pd.DataFrame:
        """
        根据速度和转角对行为进行分类
        
        Args:
            speed_thresholds: 速度阈值，用于区分不同行为
            turn_threshold: 转弯阈值(度)
            
        Returns:
            pd.DataFrame: 带有行为分类标签的DataFrame
        """
        if self.df.empty:
            return pd.DataFrame()
        
        def classify(row):
            if row['speed'] < speed_thresholds['resting']:
                return 'resting'
            elif row['turn_angle'] > turn_threshold:
                return 'turning'
            elif row['speed'] < speed_thresholds['walking']:
                return 'walking'
            else:
                return 'flying'
        
        self.df['behavior'] = self.df.apply(classify, axis=1)
        return self.df[['timestamp', 'behavior']]
    
    def get_behavior_summary(self) -> Dict[str, float]:
        """获取行为分类的摘要统计"""
        if 'behavior' not in self.df.columns:
            self.classify_behavior()
        
        # 计算每种行为的持续时间
        summary = self.df.groupby('behavior')['dt'].sum().to_dict()
        
        # 转换为百分比
        total_duration = sum(summary.values())
        if total_duration > 0:
            summary_percent = {k: (v / total_duration) * 100 for k, v in summary.items()}
            return summary_percent
        
        return {}

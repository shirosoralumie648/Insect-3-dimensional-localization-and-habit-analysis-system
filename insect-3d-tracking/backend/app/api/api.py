from fastapi import APIRouter
from .endpoints import (
    auth, users, projects, cameras, 
    apriltag, detection, analysis, 
    recording, datasets, models
)

api_router = APIRouter()

# 认证路由
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])

# 用户管理路由
api_router.include_router(users.router, prefix="/users", tags=["用户管理"])

# 项目管理路由
api_router.include_router(projects.router, prefix="/projects", tags=["项目管理"])

# 相机配置路由
api_router.include_router(cameras.router, prefix="/cameras", tags=["相机配置"])

# Apriltag检测和标定路由
api_router.include_router(apriltag.router, prefix="/apriltag", tags=["Apriltag检测"])

# 昆虫检测和三维定位路由
api_router.include_router(detection.router, prefix="/detection", tags=["昆虫检测与定位"])

# 习性分析路由
api_router.include_router(analysis.router, prefix="/analysis", tags=["习性分析"])

# 视频录制路由
api_router.include_router(recording.router, prefix="/recording", tags=["视频录制"])

# 数据集管理路由
api_router.include_router(datasets.router, prefix="/datasets", tags=["数据集管理"])

# 模型训练和推理路由
api_router.include_router(models.router, prefix="/models", tags=["模型管理"])

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import os
import socketio
from sqlalchemy.orm import Session
import uvicorn

from .config import settings
from .database.session import engine, Base, get_db
from .database import models
from .api.endpoints import cameras, apriltag, detection, analysis, recording, datasets, models as model_api, projects

# 创建数据库表
models.Base.metadata.create_all(bind=engine)

# 创建存储目录
os.makedirs(settings.STORAGE_PATH, exist_ok=True)
os.makedirs(settings.VIDEO_PATH, exist_ok=True)
os.makedirs(settings.DATASET_PATH, exist_ok=True)
os.makedirs(settings.MODEL_PATH, exist_ok=True)

# 创建Socket.IO实例用于实时通信
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description=f"{settings.APP_NAME} - 昆虫3D定位与习性分析系统",
    version=settings.APP_VERSION,
    docs_url=None,  # 自定义Swagger UI路径
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自定义Swagger UI路径
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.APP_NAME} - API文档",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

# 包含Socket.IO应用
app.mount("/ws", socket_app)

# 设置API前缀和路由
api_prefix = settings.API_PREFIX

# 注册API路由
app.include_router(cameras.router, prefix=f"{api_prefix}/cameras", tags=["相机控制"])
app.include_router(apriltag.router, prefix=f"{api_prefix}/apriltag", tags=["Apriltag检测"])
app.include_router(detection.router, prefix=f"{api_prefix}/detection", tags=["昆虫检测"])
app.include_router(analysis.router, prefix=f"{api_prefix}/analysis", tags=["习性分析"])
app.include_router(recording.router, prefix=f"{api_prefix}/recording", tags=["视频录制"])
app.include_router(datasets.router, prefix=f"{api_prefix}/datasets", tags=["数据集管理"])
app.include_router(model_api.router, prefix=f"{api_prefix}/models", tags=["模型管理"])
app.include_router(projects.router, prefix=f"{api_prefix}/projects", tags=["项目管理"])

# 根路由
@app.get("/")
async def root():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs_url": "/docs"
    }

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Socket.IO连接事件
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

# 启动应用
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)

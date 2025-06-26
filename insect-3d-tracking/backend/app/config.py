import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """应用配置设置"""
    APP_NAME: str = os.getenv("APP_NAME", "InsectTracker")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_secret_key_here_change_in_production")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")

    # JWT
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    
    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./insect_tracker.db")
    
    # 存储配置
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    VIDEO_PATH: str = os.getenv("VIDEO_PATH", "./storage/videos")
    DATASET_PATH: str = os.getenv("DATASET_PATH", "./storage/datasets")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./storage/models")
    
    # 相机设置
    DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "1280x720")
    DEFAULT_FPS: int = int(os.getenv("DEFAULT_FPS", "30"))
    
    # YOLO设置
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "./storage/models/yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    
    class Config:
        case_sensitive = True


settings = Settings()

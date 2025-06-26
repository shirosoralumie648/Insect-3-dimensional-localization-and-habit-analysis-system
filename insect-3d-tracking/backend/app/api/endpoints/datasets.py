from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Any, List
import shutil

from ...database.session import get_db
from ...database.models import User, Project, Dataset
from ...schemas.dataset import DatasetCreate, DatasetResponse, AnnotationCreate
from ...core.dataset import DatasetManager
from ..deps import get_current_active_user
from ...config import settings

router = APIRouter()

@router.post("/", response_model=DatasetResponse)
def create_dataset(
    *, 
    db: Session = Depends(get_db),
    dataset_in: DatasetCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    创建一个新的数据集记录
    """
    project = db.query(Project).filter(Project.id == dataset_in.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "没有权限在此项目下创建数据集")

    dataset = Dataset(**dataset_in.dict(), created_by=current_user.id)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    # 初始化数据集目录
    DatasetManager(dataset.name)
    
    return dataset

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    获取数据集详情
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "数据集不存在")
    
    project = db.query(Project).filter(Project.id == dataset.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "没有权限访问此数据集")
    
    return dataset

@router.post("/{dataset_id}/images")
def upload_image_to_dataset(
    dataset_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    上传单个图像到数据集
    """
    dataset = get_dataset(dataset_id, db, current_user)
    dm = DatasetManager(dataset.name)

    try:
        # 保存上传的文件
        temp_path = settings.TEMP_DIR / file.filename
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 添加到数据集
        new_image_path = dm.add_image(str(temp_path))
        temp_path.unlink() # 删除临时文件

        return {"message": "图像上传成功", "path": new_image_path}
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"上传失败: {e}")

@router.post("/{dataset_id}/annotations")
def save_dataset_annotation(
    dataset_id: int,
    annotation: AnnotationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    保存图像标注
    """
    dataset = get_dataset(dataset_id, db, current_user)
    dm = DatasetManager(dataset.name)
    
    try:
        dm.save_annotation(annotation.image_name, annotation.coco_annotation)
        return {"message": "标注保存成功"}
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"保存失败: {e}")

@router.post("/{dataset_id}/prepare_for_yolo")
def prepare_dataset_for_yolo(
    dataset_id: int,
    class_names: List[str] = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    转换标注为YOLO格式并创建配置文件
    """
    dataset = get_dataset(dataset_id, db, current_user)
    dm = DatasetManager(dataset.name)
    
    try:
        dm.convert_to_yolo()
        dm.create_yolo_config(class_names)
        return {"message": "数据集已准备好进行YOLO训练", "config_path": str(dm.config_path)}
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"准备失败: {e}")

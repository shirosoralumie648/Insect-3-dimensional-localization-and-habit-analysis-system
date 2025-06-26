from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Any

from ...database.session import get_db
from ...database.models import User, Project, Dataset, Model
from ...schemas.model import ModelCreate, ModelResponse, TrainingJobCreate, TrainingJobResponse
from ...core.dataset import DatasetManager
from ...core.training import training_manager
from ..deps import get_current_active_user

router = APIRouter()

@router.post("/train", response_model=TrainingJobResponse)
def start_new_training(
    *, 
    db: Session = Depends(get_db),
    job_in: TrainingJobCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    开始一个新的模型训练任务
    """
    dataset = db.query(Dataset).filter(Dataset.id == job_in.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    project = db.query(Project).filter(Project.id == dataset.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not enough permissions")

    dm = DatasetManager(dataset.name)
    
    if not dm.config_path.exists():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Dataset is not prepared for training. Call prepare_for_yolo first.")
        
    try:
        train_id = training_manager.start_training(
            dataset_config_path=str(dm.config_path),
            model_name=job_in.base_model,
            epochs=job_in.epochs,
            batch_size=job_in.batch_size,
            img_size=job_in.img_size,
            project_name=f"{dataset.name}_training",
            device=job_in.device
        )
        
        model = Model(
            project_id=dataset.project_id,
            dataset_id=dataset.id,
            name=f"{dataset.name}_{train_id}",
            version="1.0",
            status="training",
            train_job_id=train_id,
            parameters=job_in.dict(),
            created_by=current_user.id
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        
        return {"job_id": train_id, "model_id": model.id, "status": "started"}
    except Exception as e:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Failed to start training: {e}")

@router.get("/train/status/{job_id}")
def get_training_status(job_id: str) -> Any:
    """
    获取训练任务的状态
    """
    status = training_manager.get_training_status(job_id)
    if status is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Training job not found")
    return status

@router.get("/train/all")
def list_all_training_jobs() -> Any:
    """
    列出所有训练任务
    """
    return training_manager.list_all_trainings()

@router.post("/train/complete/{job_id}")
def complete_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    当训练完成后，由系统内部调用，更新模型状态和路径
    """
    status = training_manager.get_training_status(job_id)
    if not status or status['status'] != 'completed':
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Training job not completed or not found")
    
    model = db.query(Model).filter(Model.train_job_id == job_id).first()
    if not model:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Associated model record not found")

    project = db.query(Project).filter(Project.id == model.project_id).first()
    if not project or (project.user_id != current_user.id and not current_user.is_admin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not enough permissions to update this model")

    model.status = 'completed'
    model.path = str(status['results']['best_model_path'])
    model.results = status['results']
    db.commit()

    return {"message": "Model status updated", "model_id": model.id}

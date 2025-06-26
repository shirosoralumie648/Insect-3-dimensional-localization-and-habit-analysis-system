import json
import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.database.models import Dataset, Model

def test_start_new_training(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试开始一个新的模型训练任务。
    """
    # 模拟训练管理器
    mock_start = MagicMock(return_value="test_train_job_123")
    mocker.patch('app.core.training.training_manager.start_training', mock_start)
    
    # 模拟 DatasetManager 以避免文件系统交互
    mock_dm_instance = MagicMock()
    mock_dm_instance.config_path.exists.return_value = True
    mocker.patch('app.api.endpoints.models.DatasetManager', return_value=mock_dm_instance)

    train_data = {
        "dataset_id": test_dataset.id,
        "base_model": "yolov8n.pt",
        "epochs": 1
    }

    response = client.post("/api/models/train", json=train_data, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert content["job_id"] == "test_train_job_123"
    assert content["status"] == "started"
    assert "model_id" in content
    mock_start.assert_called_once()

def test_get_training_status(client: TestClient, auth_headers: dict, mocker) -> None:
    """
    测试获取训练任务的状态。
    """
    mock_status = MagicMock(return_value={"status": "running", "progress": 50})
    mocker.patch('app.core.training.training_manager.get_training_status', mock_status)

    job_id = "test_train_job_123"
    response = client.get(f"/api/models/train/status/{job_id}", headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert content["status"] == "running"
    assert content["progress"] == 50
    mock_status.assert_called_with(job_id)

def test_list_all_training_jobs(client: TestClient, auth_headers: dict, mocker) -> None:
    """
    测试列出所有训练任务。
    """
    mock_list = MagicMock(return_value={"test_train_job_123": {"status": "running"}})
    mocker.patch('app.core.training.training_manager.list_all_trainings', mock_list)

    response = client.get("/api/models/train/all", headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert "test_train_job_123" in content
    mock_list.assert_called_once()

def test_complete_training_job(client: TestClient, auth_headers: dict, test_dataset: Dataset, db: Session, mocker) -> None:
    """
    测试完成训练任务并更新模型记录。
    """
    job_id = "test_train_job_456"
    # 先创建一个模型记录与job_id关联
    model = Model(
        dataset_id=test_dataset.id,
        name=f"model_for_{job_id}",
        path=f"/fake/models/{job_id}",
        status="training",
        train_job_id=job_id
    )
    db.add(model)
    db.commit()

    # 模拟训练已完成
    mock_status = MagicMock(return_value={
        'status': 'completed', 
        'results': {'best_model_path': '/fake/path/best.pt'}
    })
    mocker.patch('app.core.training.training_manager.get_training_status', mock_status)

    response = client.post(f"/api/models/train/complete/{job_id}", headers=auth_headers)
    
    assert response.status_code == 200
    assert response.json()["message"] == "Model status updated"

    # 验证数据库中的模型状态已被更新
    db.refresh(model)
    assert model.status == 'completed'
    assert model.path == '/fake/path/best.pt'

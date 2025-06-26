import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.database.models import Project, Dataset

def test_create_dataset(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试创建新数据集。
    """
    # 模拟DatasetManager的__init__，避免在测试中创建真实目录
    mock_init = mocker.patch('app.core.dataset.DatasetManager.__init__', return_value=None)
    
    data = {"name": "New Test Dataset", "project_id": test_project.id}
    response = client.post("/api/datasets/", json=data, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert content["name"] == "New Test Dataset"
    assert content["project_id"] == test_project.id
    mock_init.assert_called_once()


def test_upload_image_to_dataset(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试向数据集上传图片。
    """
    # 模拟 get_dataset, 因为它在端点内部被调用
    mocker.patch('app.api.endpoints.datasets.get_dataset', return_value=test_dataset)
    
    # 模拟 DatasetManager 实例以避免文件系统操作
    mock_manager_instance = MagicMock()
    mock_manager_instance.add_image.return_value = "/fake/path/image.jpg"
    mocker.patch('app.api.endpoints.datasets.DatasetManager', return_value=mock_manager_instance)
    
    fake_image_bytes = b"fake image data"
    # 注意：FastAPI TestClient 的 files 参数的 key 应该匹配端点函数中定义的参数名 'file'
    files = {'file': ('test_image.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg')}

    response = client.post(f"/api/datasets/{test_dataset.id}/images", files=files, headers=auth_headers)
    
    assert response.status_code == 200
    # 响应体中的 key 是 'path', 不是 'file_path'
    assert response.json()["path"] == "/fake/path/image.jpg"
    mock_manager_instance.add_image.assert_called_once()

def test_save_annotation(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试保存标注信息。
    """
    mocker.patch('app.api.endpoints.datasets.get_dataset', return_value=test_dataset)
    
    # 模拟 DatasetManager 实例以避免文件系统操作
    mock_manager_instance = MagicMock()
    mocker.patch('app.api.endpoints.datasets.DatasetManager', return_value=mock_manager_instance)

    anno_data = {
        "image_name": "test.jpg",
        "coco_annotation": {"bbox": [10, 20, 30, 40], "category_id": 1}
    }
    response = client.post(f"/api/datasets/{test_dataset.id}/annotations", json=anno_data, headers=auth_headers)
    
    assert response.status_code == 200
    # 端点返回成功消息，而不是 annotation_id
    assert response.json()["message"] == "标注保存成功"
    mock_manager_instance.save_annotation.assert_called_once_with("test.jpg", {"bbox": [10, 20, 30, 40], "category_id": 1})

def test_prepare_dataset_for_yolo(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试准备YOLO格式的数据集。
    """
    mocker.patch('app.api.endpoints.datasets.get_dataset', return_value=test_dataset)
    
    # 模拟 DatasetManager 实例以访问 config_path 属性并检查方法调用
    mock_dm_instance = MagicMock()
    mock_dm_instance.config_path = "/fake/path/dataset.yaml"
    mocker.patch('app.api.endpoints.datasets.DatasetManager', return_value=mock_dm_instance)

    # 端点 URL 是 /prepare_for_yolo
    # 端点需要 'class_names' 作为表单数据
    response = client.post(
        f"/api/datasets/{test_dataset.id}/prepare_for_yolo", 
        data={"class_names": ["insect"]}, 
        headers=auth_headers
    )
    
    assert response.status_code == 200
    assert "config_path" in response.json()
    assert response.json()["config_path"] == "/fake/path/dataset.yaml"
    
    # 验证两个核心方法都被调用了
    mock_dm_instance.convert_to_yolo.assert_called_once()
    mock_dm_instance.create_yolo_config.assert_called_once_with(["insect"])

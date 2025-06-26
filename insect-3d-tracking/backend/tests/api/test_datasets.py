import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.database.models import Project, Dataset

def test_create_dataset(client: TestClient, auth_headers: dict, test_project: Project) -> None:
    """
    测试创建新数据集。
    """
    data = {"name": "New Test Dataset", "project_id": test_project.id}
    with patch('app.core.dataset.DatasetManager.create_dataset_dirs') as mock_create_dirs:
        response = client.post("/api/datasets/", json=data, headers=auth_headers)
        assert response.status_code == 200
        content = response.json()
        assert content["name"] == "New Test Dataset"
        assert content["project_id"] == test_project.id
        mock_create_dirs.assert_called_once()

def test_upload_image_to_dataset(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试向数据集上传图片。
    """
    mock_save = MagicMock(return_value="/fake/path/image.jpg")
    mocker.patch('app.core.dataset.DatasetManager.save_image', mock_save)

    fake_image_bytes = b"fake image data"
    files = {'upload_file': ('test_image.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg')}

    response = client.post(f"/api/datasets/{test_dataset.id}/images", files=files, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["file_path"] == "/fake/path/image.jpg"
    mock_save.assert_called_once()

def test_save_annotation(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试保存标注信息。
    """
    mock_save_anno = MagicMock(return_value=1)
    mocker.patch('app.core.dataset.DatasetManager.save_coco_annotation', mock_save_anno)

    anno_data = {
        "image_name": "test.jpg",
        "coco_annotation": {"bbox": [10, 20, 30, 40], "category_id": 1}
    }
    response = client.post(f"/api/datasets/{test_dataset.id}/annotations", json=anno_data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["annotation_id"] == 1
    mock_save_anno.assert_called_once()

def test_prepare_dataset_for_yolo(client: TestClient, auth_headers: dict, test_dataset: Dataset, mocker) -> None:
    """
    测试准备YOLO格式的数据集。
    """
    mock_prepare = MagicMock(return_value={"config_path": "/fake/path/data.yaml"})
    mocker.patch('app.core.dataset.DatasetManager.prepare_for_yolo', mock_prepare)

    response = client.post(f"/api/datasets/{test_dataset.id}/prepare", headers=auth_headers)
    assert response.status_code == 200
    assert "config_path" in response.json()
    mock_prepare.assert_called_once()

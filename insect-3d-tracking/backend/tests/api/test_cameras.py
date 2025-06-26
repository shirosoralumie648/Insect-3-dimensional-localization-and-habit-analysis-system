from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.models import Project, CameraConfig

def test_create_camera_config(client: TestClient, auth_headers: dict, test_project: Project) -> None:
    """
    测试创建新的相机配置。
    """
    data = {
        "project_id": test_project.id,
        "camera_name": "Test Cam",
        "camera_index": 0,
        "camera_type": "USB",
        "resolution_width": 1920,
        "resolution_height": 1080,
        "fps": 30
    }
    response = client.post("/api/cameras/", json=data, headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["camera_name"] == data["camera_name"]
    assert content["project_id"] == test_project.id
    assert "id" in content

def test_get_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取单个相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, camera_name="Get Test Cam", camera_index=1)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    response = client.get(f"/api/cameras/{camera.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["camera_name"] == camera.camera_name
    assert content["id"] == camera.id

def test_get_all_camera_configs(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取一个项目的所有相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, camera_name="List Test Cam", camera_index=2)
    db.add(camera)
    db.commit()

    response = client.get(f"/api/cameras/project/{test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert isinstance(content, list)
    assert len(content) > 0

def test_update_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试更新相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, camera_name="Update Cam", camera_index=3, fps=25)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    update_data = {"fps": 60}
    response = client.put(f"/api/cameras/{camera.id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["fps"] == 60
    assert content["camera_name"] == "Update Cam"

def test_delete_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试删除相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, camera_name="Delete Cam", camera_index=4)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    response = client.delete(f"/api/cameras/{camera.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["message"] == "Camera config deleted successfully"

    deleted_cam = db.query(CameraConfig).filter(CameraConfig.id == camera.id).first()
    assert deleted_cam is None

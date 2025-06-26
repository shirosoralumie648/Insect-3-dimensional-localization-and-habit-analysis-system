from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.models import Project, CameraConfig

def test_create_camera_config(client: TestClient, auth_headers: dict, test_project: Project) -> None:
    """
    测试创建新的相机配置。
    """
    data = {
        "project_id": test_project.id,
        "name": "Test Cam",
        "camera_index": 0,
        "width": 1920,
        "height": 1080,
        "fps": 30
    }
    response = client.post("/api/cameras/", json=data, headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["name"] == data["name"]
    assert content["project_id"] == test_project.id
    assert "id" in content

def test_get_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取单个相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, name="Get Test Cam", camera_index=1, width=1280, height=720, fps=30)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    response = client.get(f"/api/cameras/{camera.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["name"] == camera.name
    assert content["id"] == camera.id

def test_get_all_camera_configs(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取一个项目的所有相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, name="List Test Cam", camera_index=2, width=1280, height=720, fps=30)
    db.add(camera)
    db.commit()

    response = client.get(f"/api/cameras/?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert isinstance(content, list)
    # The project fixture might already have configs, so just check that our new one is there
    assert any(c['name'] == "List Test Cam" for c in content)


def test_update_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试更新相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, name="Update Cam", camera_index=3, width=1280, height=720, fps=25)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    update_data = {"fps": 60}
    response = client.put(f"/api/cameras/{camera.id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["fps"] == 60
    assert content["name"] == "Update Cam"

def test_delete_camera_config(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试删除相机配置。
    """
    camera = CameraConfig(project_id=test_project.id, name="Delete Cam", camera_index=4, width=640, height=480, fps=15)
    db.add(camera)
    db.commit()
    db.refresh(camera)

    response = client.delete(f"/api/cameras/{camera.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["message"] == "Camera config deleted successfully"

    deleted_cam = db.query(CameraConfig).filter(CameraConfig.id == camera.id).first()
    assert deleted_cam is None

import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from app.database.models import Project, DetectionSession

def test_start_detection_session(client: TestClient, auth_headers: dict, test_project: Project) -> None:
    """
    测试开始一个新的检测会话。
    """
    response = client.post(f"/api/detection/session/start?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert "id" in content
    assert "project_id" in content
    assert content["project_id"] == test_project.id

def test_detect_3d(client: TestClient, auth_headers: dict, test_project: Project, db: Session, mocker) -> None:
    """
    测试三维检测端点。
    """
    # 创建一个测试会话
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    # 模拟核心检测函数
    mock_detect_3d = MagicMock(return_value=([{"x": 1, "y": 2, "z": 3}]))
    mocker.patch('app.api.endpoints.detection.Localizer3D.localize_3d', mock_detect_3d)

    # 模拟图像文件上传
    fake_image_bytes = b"fake image data"
    files = [
        ('left_image_file', ('left.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg')),
        ('right_image_file', ('right.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg'))
    ]

    response = client.post(f"/api/detection/session/{session.id}/detect", files=files, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]['x'] == 1
    mock_detect_3d.assert_called_once()

def test_get_detection_session(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取检测会话的结果。
    """
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    response = client.get(f"/api/detection/session/{session.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == session.id
    assert "results" in content

def test_delete_detection_session(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试删除一个检测会话。
    """
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    response = client.delete(f"/api/detection/session/{session.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["message"] == "Detection session and its results deleted successfully"

    deleted_session = db.query(DetectionSession).filter(DetectionSession.id == session.id).first()
    assert deleted_session is None

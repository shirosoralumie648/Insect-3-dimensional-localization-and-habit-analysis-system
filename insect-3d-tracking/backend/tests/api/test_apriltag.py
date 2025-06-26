import base64
import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.database.models import Project, ApriltagConfig, User

def test_detect_apriltags(client: TestClient, db: Session, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试Apriltag检测端点。
    """
    # 1. 创建依赖的 ApriltagConfig
    apriltag_config = ApriltagConfig(
        project_id=test_project.id,
        tag_family="tag36h11",
        tag_id=1,
        tag_size=0.05
    )
    db.add(apriltag_config)
    db.commit()
    db.refresh(apriltag_config)

    # 2. 准备请求数据
    settings_payload = {
        "apriltag_config_id": apriltag_config.id,
        "estimate_pose": False
    }
    fake_image_bytes = b"fake image data"
    base64_image_payload = base64.b64encode(fake_image_bytes).decode('utf-8')

    request_payload = {
        "settings": settings_payload,
        "base64_image": base64_image_payload
    }

    # 3. 模拟核心检测函数
    mock_detection_result = MagicMock()
    mock_detection_result.tag_id = 1
    mock_detection_result.center = (100, 100)
    mock_detection_result.corners = [(50, 50), (150, 50), (150, 150), (50, 150)]
    mock_detection_result.hamming = 0
    mock_detection_result.decision_margin = 100.0
    
    mock_detect = MagicMock(return_value=[mock_detection_result])
    mocker.patch('app.api.endpoints.apriltag.ApriltagDetector.detect', mock_detect)

    # 4. 发送请求
    response = client.post("/api/apriltag/detect", json=request_payload, headers=auth_headers)

    # 5. 断言
    assert response.status_code == 200, response.text
    content = response.json()
    assert "detections" in content
    assert isinstance(content["detections"], list)
    assert len(content["detections"]) == 1
    assert content["detections"][0]["tag_id"] == 1
    mock_detect.assert_called_once()

def test_calibrate_camera(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试相机标定端点。
    """
    # 模拟核心标定函数
    mock_calibrate = MagicMock(return_value=({"mtx": [[1,0,0],[0,1,0],[0,0,1]], "dist": [0,0,0,0,0]}, 0.1))
    mocker.patch('app.api.endpoints.apriltag.calibrate_camera', mock_calibrate)

    # 模拟图像文件上传
    fake_image_bytes = b"fake image data"
    files = [('files', ('calib_1.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg')),
             ('files', ('calib_2.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg'))]
    
    data = {"marker_size_m": 0.025, "grid_width": 8, "grid_height": 6}
    
    response = client.post(f"/api/apriltag/calibrate?project_id={test_project.id}", data=data, files=files, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert "calibration_data" in content
    assert "reprojection_error" in content
    assert "mtx" in content["calibration_data"]
    mock_calibrate.assert_called_once()

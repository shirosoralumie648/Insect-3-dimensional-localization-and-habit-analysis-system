import base64
import json
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.database.models import Project, ApriltagConfig, User, CameraConfig

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
    # 使用一个真实的、小的 base64 编码图像
    import numpy as np
    import cv2
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    payload = {
        "image_base64": img_base64,
        "settings": {
            "apriltag_config_id": apriltag_config.id,
            "nthreads": 1,
            "quad_decimate": 2.0,
            "quad_sigma": 0.0,
            "refine_edges": 1,
            "decode_sharpening": 0.25,
            "debug": 0,
            "draw_detections": False
        }
    }

    # 3. 模拟检测器
    mock_detector_instance = MagicMock()
    mock_detection = MagicMock()
    mock_detection.tag_id = 1
    mock_detection.center = np.array([5, 5])
    mock_detection.corners = np.array([[0,0], [10,0], [10,10], [0,10]])
    mock_detection.hamming = 0
    mock_detection.decision_margin = 100.0
    mock_detector_instance.detect.return_value = [mock_detection]
    mocker.patch('app.api.endpoints.apriltag.ApriltagDetector', return_value=mock_detector_instance)

    # 4. 发送请求
    response = client.post("/api/apriltag/detect", json=payload, headers=auth_headers)

    # 5. 断言
    assert response.status_code == 200, response.text
    content = response.json()
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["tag_id"] == 1
    assert content[0]["center"] == [5.0, 5.0]
    mock_detector_instance.detect.assert_called_once()

def test_calibrate_camera(client: TestClient, db: Session, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试相机标定端点。
    """
    # 1. 创建依赖项
    camera_config = CameraConfig(
        project_id=test_project.id, camera_index=0, width=1920, height=1080, fps=30
    )
    apriltag_config = ApriltagConfig(
        project_id=test_project.id, tag_family="tag36h11", tag_id=1, tag_size=0.05
    )
    db.add_all([camera_config, apriltag_config])
    db.commit()
    db.refresh(camera_config)
    db.refresh(apriltag_config)

    # 2. 准备请求数据
    settings_payload = {
        "camera_config_id": camera_config.id,
        "apriltag_config_id": apriltag_config.id,
        "num_images": 2, # Use a small number for testing
        "min_images": 1,
        "capture_delay": 0.1
    }

    # 3. 模拟外部依赖
    mock_camera = MagicMock()
    mock_camera.is_running = True
    mock_camera.open.return_value = True
    mock_camera.get_frame.return_value = (True, "fake_frame_data")
    mocker.patch('app.api.endpoints.apriltag.get_camera_instance', return_value=mock_camera)

    mock_detector_instance = MagicMock()
    mock_detection = MagicMock()
    mock_detector_instance.detect.return_value = [mock_detection]
    mocker.patch('app.api.endpoints.apriltag.ApriltagDetector', return_value=mock_detector_instance)

    mock_calibrate = mocker.patch('app.api.endpoints.apriltag.calibrate_camera', 
                                  return_value=({}, 0.5, [], [], []))
    # This is the core function to mock
    mocker.patch('app.core.apriltag.calibrate_camera', 
                 return_value=({'mtx': 'fake_mtx', 'dist': 'fake_dist'}, 0.5))

    # 4. 发送请求
    response = client.post("/api/apriltag/calibrate", json=settings_payload, headers=auth_headers)

    # 5. 断言
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["success"] is True
    assert content["camera_config_id"] == camera_config.id
    assert "calibration_data" in content
    assert content["calibration_data"]["mtx"] == 'fake_mtx'
    assert content["reprojection_error"] == 0.5
    mock_calibrate.assert_called_once()

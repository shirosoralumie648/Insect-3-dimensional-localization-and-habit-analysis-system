import io
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.database.models import Project

def test_detect_apriltags(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试Apriltag检测端点。
    """
    # 模拟核心检测函数
    mock_detect = MagicMock(return_value=([{"id": 0, "corners": [[0,0], [1,0], [1,1], [0,1]]}], None))
    mocker.patch('app.core.apriltag.ApriltagDetector.detect', mock_detect)

    # 模拟一个图像文件上传
    fake_image_bytes = b"fake image data"
    files = {'file': ('test_image.jpg', io.BytesIO(fake_image_bytes), 'image/jpeg')}
    
    response = client.post(f"/api/apriltag/detect?project_id={test_project.id}", files=files, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert "tags" in content
    assert isinstance(content["tags"], list)
    assert len(content["tags"]) == 1
    assert content["tags"][0]["id"] == 0
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

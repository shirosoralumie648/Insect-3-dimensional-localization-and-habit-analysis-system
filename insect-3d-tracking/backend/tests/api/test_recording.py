import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from app.database.models import Project, RecordingSettings, Video

def test_recording_workflow(client: TestClient, db: Session, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试完整的录制工作流程：创建设置 -> 开始 -> 状态 -> 停止 -> 列表 -> 删除
    """
    # 1. 创建录制设置
    settings_data = {
        "project_id": test_project.id,
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "fourcc": "mp4v",
        "output_dir": "/tmp/videos",
        "filename_prefix": "test_vid"
    }
    response = client.post("/api/recording/settings/", json=settings_data, headers=auth_headers)
    assert response.status_code == 200, response.text
    settings = response.json()
    settings_id = settings["id"]
    assert settings["width"] == 1920

    # 2. 读取录制设置
    response = client.get(f"/api/recording/settings/{settings_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["fps"] == 30

    camera_index = 0

    # 模拟 VideoRecorder
    mock_recorder_instance = MagicMock()
    mock_recorder_instance.get_status.return_value = {'is_recording': True}
    mock_recorder_instance.stop.return_value = {
        'output_path': '/tmp/videos/test.mp4',
        'duration': 10.5,
    }
    mock_recorder_instance.frame_size = (1920, 1080)
    mock_recorder_instance.fps = 30.0
    mocker.patch('app.api.endpoints.recording.get_recorder_instance', return_value=mock_recorder_instance)
    mocker.patch('app.api.endpoints.recording.recorders', {camera_index: mock_recorder_instance})

    # 3. 开始录制
    response = client.post(f"/api/recording/start/{camera_index}?settings_id={settings_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["message"] == "录制已开始"
    mock_recorder_instance.start.assert_called_once()

    # 4. 获取录制状态
    response = client.get(f"/api/recording/status/{camera_index}", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert response.json()["is_recording"] is True

    # 5. 停止录制
    response = client.post(f"/api/recording/stop/{camera_index}", headers=auth_headers)
    assert response.status_code == 200, response.text
    video_data = response.json()
    video_id = video_data["id"]
    assert video_data["name"] == "test.mp4"

    # 6. 列出视频
    response = client.get(f"/api/recording/videos/?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    videos = response.json()
    assert len(videos) > 0
    assert videos[0]["name"] == "test.mp4"

    # 7. 删除视频
    mock_os_path_exists = mocker.patch('os.path.exists', return_value=True)
    mock_os_remove = mocker.patch('os.remove')
    response = client.delete(f"/api/recording/videos/{video_id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    mock_os_path_exists.assert_called_once()
    mock_os_remove.assert_called_once()

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.database.models import Project

def test_update_and_get_recording_settings(client: TestClient, auth_headers: dict, test_project: Project) -> None:
    """
    测试更新和获取录制设置。
    """
    settings_data = {
        "resolution": "1080p",
        "fps": 30,
        "codec": "mp4v"
    }
    response = client.put(f"/api/recording/settings?project_id={test_project.id}", json=settings_data, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["resolution"] == "1080p"

    response = client.get(f"/api/recording/settings?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["fps"] == 30

def test_start_recording(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试开始录制。
    """
    mock_start = MagicMock()
    mocker.patch('app.api.endpoints.recording.VideoRecorder.start', mock_start)

    response = client.post(f"/api/recording/start?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "Recording started for project Test Project"
    mock_start.assert_called_once()

def test_stop_recording(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试停止录制。
    """
    mock_stop = MagicMock()
    mocker.patch('app.api.endpoints.recording.VideoRecorder.stop', mock_stop)

    response = client.post(f"/api/recording/stop?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "Recording stopped for project Test Project"
    mock_stop.assert_called_once()

def test_get_recording_status(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试获取录制状态。
    """
    mock_status = MagicMock(return_value={"is_recording": True})
    mocker.patch('app.api.endpoints.recording.VideoRecorder.get_status', mock_status)

    response = client.get(f"/api/recording/status?project_id={test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["is_recording"] is True
    mock_status.assert_called_once()

def test_list_videos(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试列出所有录制的视频。
    """
    with patch('os.path.exists', return_value=True), \
         patch('os.listdir', return_value=["video1.mp4", "video2.avi"]):
        response = client.get(f"/api/recording/videos?project_id={test_project.id}", headers=auth_headers)
        assert response.status_code == 200
        content = response.json()
        assert isinstance(content, list)
        assert len(content) == 2
        assert "video1.mp4" in [v['filename'] for v in content]

def test_delete_video(client: TestClient, auth_headers: dict, test_project: Project, mocker) -> None:
    """
    测试删除一个视频。
    """
    with patch('os.path.exists', return_value=True), \
         patch('os.remove') as mock_remove:
        response = client.delete(f"/api/recording/videos/video1.mp4?project_id={test_project.id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["message"] == "Video video1.mp4 deleted successfully"
        mock_remove.assert_called_once()

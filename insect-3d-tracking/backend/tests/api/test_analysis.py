from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import MagicMock
from datetime import datetime

from app.database.models import Project, DetectionSession, AnalysisResult, User, TrackingResult

def test_run_analysis(client: TestClient, auth_headers: dict, test_project: Project, db: Session, mocker, test_user: User) -> None:
    """
    测试运行一个新的习性分析。
    """
    # 创建一个测试用的检测会话
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    # 创建依赖的 TrackingResult 数据
    tracking_result = TrackingResult(
        session_id=session.id,
        timestamp=datetime.utcnow(),
        position_3d={"x": 1, "y": 2, "z": 3},
        class_name="insect"
    )
    db.add(tracking_result)
    db.commit()

    # 模拟分析器的核心方法
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.get_basic_stats', return_value={"average_speed": 0.1})
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.get_activity_timeline', return_value={})
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.get_spatial_heatmap', return_value={})
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.classify_behavior', return_value=None)
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.get_behavior_summary', return_value={})

    settings_payload = {
        "activity_time_interval": 60,
        "heatmap_grid_size": 50,
        "heatmap_projection_plane": "xy",
        "behavior_speed_thresholds": {"walking": 0.01, "running": 0.1},
        "behavior_turn_threshold": 45.0
    }
    response = client.post(f"/api/analysis/sessions/{session.id}/analyze", json=settings_payload, headers=auth_headers)
    
    assert response.status_code == 200, response.text
    content = response.json()
    assert "id" in content
    assert content["session_id"] == session.id
    assert "trajectory_stats" in content
    assert content["trajectory_stats"]["average_speed"] == 0.1

def test_get_analysis_result(client: TestClient, auth_headers: dict, test_project: Project, db: Session, test_user: User) -> None:
    """
    测试获取分析结果。
    """
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    
    analysis = AnalysisResult(
        session_id=session.id, 
        created_by=test_user.id, 
        trajectory_stats={"average_speed": 0.2},
        settings={},
        activity_timeline={},
        spatial_heatmap={},
        behavior_summary=[]
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    response = client.get(f"/api/analysis/results/{analysis.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["id"] == analysis.id
    assert content["trajectory_stats"]["average_speed"] == 0.2

def test_delete_analysis_result(client: TestClient, auth_headers: dict, test_project: Project, db: Session, test_user: User) -> None:
    """
    测试删除分析结果。
    """
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    
    analysis = AnalysisResult(
        session_id=session.id, 
        created_by=test_user.id, 
        settings={'param': 'value'},
        trajectory_stats={},
        activity_timeline={},
        spatial_heatmap={},
        behavior_summary=[]
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    response = client.delete(f"/api/analysis/results/{analysis.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    content = response.json()
    assert content["id"] == analysis.id

    deleted_analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis.id).first()
    assert deleted_analysis is None

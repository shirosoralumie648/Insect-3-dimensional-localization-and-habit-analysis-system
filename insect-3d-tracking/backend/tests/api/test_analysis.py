from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import MagicMock

from app.database.models import Project, DetectionSession, AnalysisResult

def test_run_analysis(client: TestClient, auth_headers: dict, test_project: Project, db: Session, mocker) -> None:
    """
    测试运行一个新的习性分析。
    """
    # 创建一个测试用的检测会话
    session = DetectionSession(project_id=test_project.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    # 模拟分析器的核心方法
    mock_run = MagicMock(return_value={"speed": {"mean": 0.1}})
    mocker.patch('app.core.analysis.TrajectoryAnalyzer.run_analysis', mock_run)

    data = {"detection_session_id": session.id}
    response = client.post("/api/analysis/run", json=data, headers=auth_headers)
    
    assert response.status_code == 200
    content = response.json()
    assert "id" in content
    assert content["detection_session_id"] == session.id
    assert "results" in content
    assert content["results"]["speed"]["mean"] == 0.1
    mock_run.assert_called_once()

def test_get_analysis_result(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试获取分析结果。
    """
    session = DetectionSession(project_id=test_project.id)
    analysis = AnalysisResult(detection_session=session, results={"speed": {"mean": 0.2}})
    db.add_all([session, analysis])
    db.commit()
    db.refresh(analysis)

    response = client.get(f"/api/analysis/{analysis.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == analysis.id
    assert content["results"]["speed"]["mean"] == 0.2

def test_delete_analysis_result(client: TestClient, auth_headers: dict, test_project: Project, db: Session) -> None:
    """
    测试删除分析结果。
    """
    session = DetectionSession(project_id=test_project.id)
    analysis = AnalysisResult(detection_session=session, results={})
    db.add_all([session, analysis])
    db.commit()
    db.refresh(analysis)

    response = client.delete(f"/api/analysis/{analysis.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["message"] == "Analysis result deleted successfully"

    deleted_analysis = db.query(AnalysisResult).filter(AnalysisResult.id == analysis.id).first()
    assert deleted_analysis is None

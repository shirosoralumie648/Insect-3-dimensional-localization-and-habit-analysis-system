from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.models import User, Project

def test_create_project(client: TestClient, auth_headers: dict) -> None:
    """
    测试创建一个新项目。
    """
    project_data = {
        "name": "New Test Project",
        "description": "This is a new test project."
    }
    response = client.post("/api/projects/", json=project_data, headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["name"] == project_data["name"]
    assert content["description"] == project_data["description"]
    assert "id" in content
    assert "user_id" in content

def test_get_projects(client: TestClient, auth_headers: dict, test_user: User, db: Session) -> None:
    """
    测试获取用户的所有项目。
    """
    # 先创建一个项目用于测试
    project = Project(name="List Test Project", description="...", user_id=test_user.id)
    db.add(project)
    db.commit()

    response = client.get("/api/projects/", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert isinstance(content, dict)
    assert "items" in content
    assert "total" in content
    assert isinstance(content["items"], list)
    assert len(content["items"]) > 0
    assert content["items"][0]["name"] == "List Test Project"

def test_get_project_by_id(client: TestClient, auth_headers: dict, test_user: User, db: Session) -> None:
    """
    测试通过ID获取单个项目。
    """
    project = Project(name="Get By ID Test", description="...", user_id=test_user.id)
    db.add(project)
    db.commit()
    db.refresh(project)

    response = client.get(f"/api/projects/{project.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["name"] == project.name
    assert content["id"] == project.id

def test_update_project(client: TestClient, auth_headers: dict, test_user: User, db: Session) -> None:
    """
    测试更新一个项目。
    """
    project = Project(name="Update Test", description="Before update", user_id=test_user.id)
    db.add(project)
    db.commit()
    db.refresh(project)

    update_data = {"name": "Updated Name", "description": "After update"}
    response = client.put(f"/api/projects/{project.id}", json=update_data, headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["name"] == update_data["name"]
    assert content["description"] == update_data["description"]

def test_delete_project(client: TestClient, auth_headers: dict, test_user: User, db: Session) -> None:
    """
    测试删除一个项目。
    """
    project = Project(name="Delete Test", description="...", user_id=test_user.id)
    db.add(project)
    db.commit()
    db.refresh(project)

    response = client.delete(f"/api/projects/{project.id}", headers=auth_headers)
    assert response.status_code == 200
    content = response.json()
    assert content["message"] == "Project deleted successfully"

    # 验证项目确实已被删除
    deleted_project = db.query(Project).filter(Project.id == project.id).first()
    assert deleted_project is None

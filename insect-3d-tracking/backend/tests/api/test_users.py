from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.schemas.user import UserCreate

def test_create_user(client: TestClient, db: Session) -> None:
    """
    测试创建新用户。
    """
    data = {"username": "newuser", "email": "newuser@example.com", "password": "newpassword"}
    response = client.post("/api/users/", json=data)
    assert response.status_code == 200
    content = response.json()
    assert content["username"] == data["username"]
    assert content["email"] == data["email"]
    assert "id" in content

def test_login_for_access_token(client: TestClient, test_user) -> None:
    """
    测试用户登录并获取访问令牌。
    """
    login_data = {
        "username": test_user.username,
        "password": "testpassword",
    }
    response = client.post("/api/auth/token", data=login_data)
    assert response.status_code == 200
    tokens = response.json()
    assert "access_token" in tokens
    assert tokens["token_type"] == "bearer"

def test_read_current_user(client: TestClient, auth_headers: dict) -> None:
    """
    测试获取当前登录用户的信息。
    """
    response = client.get("/api/users/me", headers=auth_headers)
    assert response.status_code == 200
    user = response.json()
    assert user["username"] == "testuser"
    assert user["email"] == "test@example.com"
    assert user["is_active"] is True

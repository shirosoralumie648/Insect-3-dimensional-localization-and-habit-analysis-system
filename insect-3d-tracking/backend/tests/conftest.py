import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from datetime import timedelta

from app.main import app
from app.database.session import get_db
from app.database.models import Base, User, Project, Dataset
from app.core.security import create_access_token, get_password_hash

# 使用内存中的SQLite数据库进行测试
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def db_engine():
    """
    为整个测试会话设置和拆卸数据库的Fixture。
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db() -> Generator:
    """
    为每个测试函数提供数据库会话的Fixture。
    在每次测试后回滚事务。
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db: Session) -> Generator:
    """
    提供一个使用测试数据库的TestClient实例的Fixture。
    """
    def override_get_db():
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
def test_user(db: Session) -> User:
    """
    在数据库中创建一个测试用户的Fixture。
    """
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=get_password_hash("testpassword"),
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_token(test_user: User) -> str:
    """
    为测试用户生成认证令牌的Fixture。
    """
    access_token_expires = timedelta(minutes=30)
    return create_access_token(subject=test_user.id, expires_delta=access_token_expires)


@pytest.fixture(scope="function")
def auth_headers(auth_token: str) -> dict:
    """
    为API请求提供认证头的Fixture。
    """
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture(scope="function")
def test_project(db: Session, test_user: User) -> "Project":
    """
    在数据库中创建一个测试项目的Fixture。
    """
    project = Project(
        name="Test Project",
        description="A project for testing purposes",
        user_id=test_user.id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@pytest.fixture(scope="function")
def test_dataset(db: Session, test_project: Project) -> Dataset:
    """
    在数据库中创建一个测试数据集的Fixture。
    """
    dataset = Dataset(
        name="Test Dataset",
        project_id=test_project.id,
        path=f"/tmp/datasets/test_dataset",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset

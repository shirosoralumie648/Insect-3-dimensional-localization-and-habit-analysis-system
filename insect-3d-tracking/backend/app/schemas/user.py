from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模型"""
    username: str
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    is_admin: Optional[bool] = False


class UserCreate(UserBase):
    """用户创建模型"""
    password: str


class UserUpdate(BaseModel):
    """用户更新模型"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None


class UserInDB(UserBase):
    """数据库中的用户模型"""
    id: int
    created_at: datetime
    hashed_password: str

    class Config:
        orm_mode = True


class User(BaseModel):
    """用户响应模型（不包含敏感信息）"""
    id: int
    username: str
    email: Optional[EmailStr] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        orm_mode = True


class UserList(BaseModel):
    """用户列表响应模型"""
    total: int
    items: List[User]


class Token(BaseModel):
    """令牌模型"""
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """令牌载荷"""
    sub: Optional[int] = None
    exp: Optional[int] = None


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""
    current_password: str
    new_password: str

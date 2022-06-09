#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/6/5 23:23
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, status, Security
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from textMG.APIs.db.db_models import User, TokenUser
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import json
from jose import jwt, JWTError

from textMG.APIs.db.database import Database
from textMG.APIs.api_loggers.api_logger import logger
from textMG.APIs.base_models.inquiries_model import Token, TokenData, UserUpdateRequest, UserRequest, UserResponse
from sqlalchemy import and_, desc


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

database = Database()
engine = database.get_db_connection()


# get SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES from db
def get_secretKey():
    session = database.get_db_session(engine)
    try:
        token_user = session.query(TokenUser).filter(TokenUser.uid == 1).one()
        return token_user
    except Exception:
        logger.critical('token can not be retrieved from db exception', exc_info=1)
        raise HTTPException(status_code=400, detail="token can not be retrieved from db")
token_user = get_secretKey()


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, token_user.SECRET_KEY, algorithm=token_user.ALGORITHM)
    return encoded_jwt


def get_user(username: str) -> User:
    """retrieve an user from db"""
    session = database.get_db_session(engine)
    try:
        user = session.query(User).filter(
            and_(User.first_name == username, User.is_active == True)).one()
        return user
    except Exception:
        logger.critical('user Not found exception', exc_info=1)
        raise HTTPException(status_code=400, detail="user Not found")


def authenticate_user(username: str, password: str) -> User:
    user = get_user(username)
    if not user:
        logger.critical(HTTPException(status_code=400, detail="user does not exist"))
        raise HTTPException(status_code=400, detail="user does not exist")
    if not verify_password(password, user.hashed_password):
        logger.critical(HTTPException(status_code=400, detail="Incorrect username or password"))
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, token_user.SECRET_KEY, algorithms=[token_user.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except ValidationError:
        logger.critical('ValidationError', exc_info=1)
        raise ValidationError
    except JWTError:
        logger.critical('JWTError', exc_info=1)
        raise JWTError
    user = get_user(username=token_data.username)
    if user is None:
        logger.critical('user does not exist')
        raise HTTPException(status_code=400, detail="user does not exist")
    if not user.is_active:
        logger.critical('Inactive user')
        raise HTTPException(status_code=400, detail="Inactive user")
    user_dict = {"id": user.id,
                 "first_name": user.first_name,
                 "last_name": user.last_name,
                 "email": user.email,
                 "is_active": user.is_active}
    return UserResponse(**user_dict)


# async def get_current_active_user(current_user: UserResponse = Security(get_current_user)) -> UserResponse:
#     if not current_user.is_active:
#         logger.critical('Inactive user')
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

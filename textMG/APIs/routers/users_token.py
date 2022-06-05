#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/6/5 23:23
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import Optional
from textMG.APIs.db.db_models import User
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import json
from jose import jwt, JWTError

from textMG.APIs.db.database import Database
from textMG.APIs.api_loggers.api_logger import logger
from sqlalchemy import and_, desc

# run the following on terminal to generate a secret key
# openssl rand -hex 32
SECRET_KEY = "ec039bcb8eada04c0de3cea9ced7440b949f6e2dadafd61ae1f8b687c214b4a9"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

crypt_context = CryptContext(schemes=["sha256_crypt", "md5_crypt"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

database = Database()
engine = database.get_db_connection()


router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


def get_password_hash(password):
    return crypt_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password, hashed_password):
    return crypt_context.verify(plain_password, hashed_password)

def get_user(username: str):
    """retrieve an user from db"""
    session = database.get_db_session(engine)
    try:
        user = session.query(User).filter(
            and_(User.first_name == username, User.is_active == True)).one()
        return user
    except Exception as e:
        logger.critical('user Not found exception', exc_info=1)

def authenticate(username, password):
    try:
        user = get_user(username)
        password_check = verify_password(password, user.hashed_password)
        return password_check
    except User.DoesNotExist:
        return False

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/token" , response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    if authenticate(username, password):
        access_token = create_access_token(
            data={"sub": username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password")
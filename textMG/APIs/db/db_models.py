#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/6/5 14:37
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, INTEGER, String, TIMESTAMP, BIGINT, BOOLEAN, text, ForeignKey

Base = declarative_base()


class User(Base):
    __tablename__ = "users_v"
    id = Column(INTEGER, primary_key=True, index=True)
    first_name = Column(String(512), nullable=True, index=True)
    last_name = Column(String(512), nullable=True, index=True)
    email = Column(String(32), unique=True, index=True)
    hashed_password = Column(String(128), nullable=True)
    is_active = Column(BOOLEAN, default=True)
    created_at = Column(TIMESTAMP, nullable=False,
                        server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, nullable=True,
                        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class TokenUser(Base):
    __tablename__ = "token_users_v"
    id = Column(INTEGER, primary_key=True, index=True)
    SECRET_KEY = Column(String(512), nullable=True)
    ALGORITHM = Column(String(32), nullable=True)
    ACCESS_TOKEN_EXPIRE_MINUTES = Column(INTEGER, nullable=True)
    ACCESS_TOKEN = Column(String(512), nullable=True)
    uid = Column(INTEGER, ForeignKey("users_v.id"))
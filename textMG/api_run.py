#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/1 20:36
"""
import uvicorn
from fastapi import FastAPI, Request
from APIs.routers.tensorflow_service import router as tf_router
from APIs.routers.health_check import router as h_router
from APIs.routers.data_generator import router as data_router
from APIs.api_loggers.api_logger import logger
from textMG.configs.config import args

from textMG.APIs.db.database import Database
from APIs.db.db_models import Base
from APIs.routers.users import router as users_router

from time import time
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from textMG.APIs.base_models.inquiries_model import Token, TokenData, UserUpdateRequest, UserRequest
from fastapi import Depends, status, Security
from datetime import datetime, timedelta
from textMG.APIs.routers.users_token import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

database = Database()
engine = database.get_db_connection()


# create db
def init_db():
    Base.metadata.create_all(engine)


# drop db
def drop_db():
    Base.metadata.drop_all(engine)


app = FastAPI()
app.include_router(h_router)
app.include_router(tf_router)
app.include_router(data_router)
app.include_router(users_router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    process_time = time() - start_time
    response.headers["Response-time"] = str(process_time)
    return response


@app.post("/token", response_model=Token, tags=['Token'])
async def access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    '''token used for data transfer'''
    user = authenticate_user(form_data.username, form_data.password)
    access_token = create_access_token(
        data={"sub": user.first_name}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}


if __name__ == '__main__':
    # drop_db()
    # init_db()
    logger.debug("the app will be started on port {}".format(args.port))
    uvicorn.run("api_run:app", host=args.host, port=args.port, workers=2,
                log_level="debug")

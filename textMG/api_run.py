#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/1 20:36
"""
import uvicorn
from fastapi import FastAPI, APIRouter
from APIs.routers.tensorflow_service import router as tf_router
from APIs.routers.health_check import router as h_router
from APIs.routers.data_generator import router as data_router
from APIs.api_loggers.api_logger import logger
from textMG.configs.config import args

app = FastAPI()

app.include_router(h_router)
app.include_router(tf_router)
app.include_router(data_router)

if __name__ == '__main__':
    logger.debug("the app will be started on port {}".format(args.port))
    uvicorn.run("api_run:app", host=args.host, port=args.port, workers=2, log_level="debug")


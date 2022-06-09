#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/2 10:39
"""
from fastapi import APIRouter, Request
from textMG.APIs.api_loggers.api_logger import logger
from fastapi.templating import Jinja2Templates

template = Jinja2Templates("/home/projects/TFpackageText/textMG/APIs/htmls")

router = APIRouter(
    tags=["health_check"],
)


@router.get("/")
async def health(req: Request):
    """get the tf API health status"""
    logger.info("health check api is called")
    return template.TemplateResponse("health_check.html", context={"request":req})

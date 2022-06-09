#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/3 17:32
"""
from fastapi import APIRouter, Depends
from textMG.datasets.generator import Generator
from textMG.APIs.base_models.responses import Response
from textMG.APIs.routers.users_token import get_current_user
from typing import Dict, Any

# load data from disk and generator.
generator = Generator()

# load batch data from disk, not from memory, and tokenize and generate it.
router = APIRouter(
    prefix="/api/v1/models",
    tags=["batch loader"],
    dependencies=[Depends(get_current_user)]
)


@router.get("/batch_load/{batch_size}")
def batch_loader(batch_size: int) -> Dict[str, Any]:
    """load data in defined patches"""
    data = {}
    iter = generator.get_next_patch(batch=batch_size)
    el = next(iter)
    data["batch"] = el
    data["input_shape"] = [len(el['x_input']), len(el['x_input'][0])]
    data["output_shape"] = len(el['y_output'])
    return Response(data=data, response_code=200, message="success", error=False)

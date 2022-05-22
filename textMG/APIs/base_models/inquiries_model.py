#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/2 20:42
"""
from pydantic import BaseModel, Field
from typing import Optional, List

class InqueryModel(BaseModel):
    query: List[str] = Field(..., min_items=1, max_items=20, min_length=2, max_length=50,
                               description="must be a List, requirements: 'min_items=1, max_items=20, min_length=2, max_length=50'")
    # description: Optional[str] = Field(
    #     None, title="The description of the inquiry", max_length=50
    # )

class InferModel(BaseModel):
    batch_infer: List[str] = Field(..., min_items=1, max_items=1000, min_length=2, max_length=50,
                               description="must be a List, requirements: 'min_items=1, max_items=20, min_length=2, max_length=50'")


class DataBatchModel(BaseModel):
    batch_size: int = Field(..., gt=0, description="the batch size")
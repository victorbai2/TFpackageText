#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/2 20:42
"""
from pydantic import BaseModel, Field, EmailStr
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


class UserRequest(BaseModel):
    first_name: str = Field(
        None, title="user first Name", max_length=50
    )
    last_name: str = Field(
        None, title="user last Name", max_length=50
    )
    email: EmailStr = Field(None, title="user Email")
    password: str = Field(
        None, title="user password", max_length=128
    )
    is_active: bool = Field(
        True, description="Value must be either True or False")


class UserUpdateRequest(BaseModel):
    id: int
    first_name: str = Field(
        None, title="user first Name", max_length=50
    )
    last_name: str = Field(
        None, title="user last Name", max_length=50
    )
    email: EmailStr = Field(None, title="user Email")
    is_active: bool = Field(
        True, description="Value must be either True or False")
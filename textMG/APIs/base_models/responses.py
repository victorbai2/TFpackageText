#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/3 00:58
"""
import json
from typing import Union, List, Dict, Any


def Response(data: Union[List, Dict[str, Any], str], response_code: int, message: str, error: Union[bool, str]) \
        -> Dict[str, Any]:
    res = {
        "data": data,
        "code": response_code,
        "message": message,
        "error": error
    }
    return res

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/3 00:58
"""
import json

def Response(data, response_code, message, error):
    res = {
        "data": data,
        "code": response_code,
        "message": message,
        "error": error
    }
    return res

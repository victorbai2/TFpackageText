#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/2 22:10
"""

def predToResult(prediction, label_dict):
    result = []
    for pred in prediction:
        for key, value in label_dict.items():
            if value == pred:
                result.append(key)
    return result
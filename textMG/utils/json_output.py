#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/2 22:43
"""
# label_dict = {'car':0, 'entertainment':1, 'military':2, 'sports':3, 'technology':4}
# time_used = 0.3
# categories=['sports',"entertainment"]
# pred_logits=[[0.18068056,0.17095093,0.22212254,0.2276705 ,0.19857548],
#             [0.18068056,0.17095093,0.22212254,0.2276705 ,0.19857548]]

import json
def jsonOutput(categories, pred_logits, label_dict, time_used):
    pred_result = {}
    __categories = {}
    for i in range(len(categories)):
        _category = {}
        _category["category"] = categories[i]
        _category["label"] = str(label_dict[categories[i]])
        _category["pred_logits"] = str(pred_logits[i])

        __categories[str(i)] = _category

    pred_result["records"] = __categories
    pred_result["time_used"] = str(time_used)

    return pred_result

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/1 22:01
"""
from fastapi import APIRouter, Request, BackgroundTasks
from textMG.APIs.api_loggers.api_logger import logger
from fastapi.templating import Jinja2Templates
from textMG.APIs.base_models.inquiries_model import InqueryModel, InferModel
from textMG.APIs.base_models.responses import Response
from textMG.configs.config import args

from textMG.main_multiGPU import do_train, do_eval, Predict
from textMG.tf_serving.grpc_infer import pred_func

template = Jinja2Templates("/home/projects/TFpackageText/textMG/APIs/htmls")
router = APIRouter(
    prefix="/api/v1",
    tags=["api/v1"],
)


@router.get("/")
async def index(req:Request):
    """load the API welcome index page"""
    logger.debug("index api is called")
    return template.TemplateResponse("index.html", context={"request":req})

@router.post("/train")
async def train():
    """call do_train() function"""
    logger.debug("train api is called, the model training is started")
    train_result = do_train()
    return Response(data=train_result, response_code=200, message="success", error=False)

@router.post("/evel")
def evaluate():
    """call do_eval function"""
    logger.debug("evaluate api is called, the evaluation is started")
    eval_result = do_eval()
    return Response(data=eval_result, response_code=200, message="success", error=False)

@router.post("/pred")
def prediction(req: InqueryModel):
    """call do_pred function"""
    logger.debug("prediction api is called, the prediction is started")
    logger.debug("inquiries is: {}".format(req))
    print("inquiries is: {}".format(req))
    pred = Predict()
    pred.do_pred(req.query)
    # task.add_task(pred.do_pred, req.inquiry)
    logger.debug(pred.get_pred_result())
    print("******" * 15)
    return Response(data=pred.get_pred_result(), response_code=200, message="success", error=False)

@router.post("/infer")
def inference(req: InferModel):
    """call infer function"""
    logger.debug("inference api is called, the inference is started")
    logger.debug("batch_infer is: {}".format(req))
    print("batch_infer is: {}".format(req))
    result = pred_func(InferModel.batch_infer, args.server, args.concurrency, args.num_tests)
    logger.debug(result)
    print("******" * 15)
    return Response(data=result, response_code=200, message="success", error=False)
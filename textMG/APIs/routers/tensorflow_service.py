#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/1 22:01
"""
import sys
sys.path.append('..')
from fastapi import APIRouter, Request, BackgroundTasks, Depends
from textMG.APIs.api_loggers.api_logger import logger
from fastapi.templating import Jinja2Templates
from textMG.APIs.base_models.inquiries_model import InqueryModel, InferModel
from textMG.APIs.base_models.responses import Response
from textMG.configs.config import args
from textMG.datasets.dataset import Dataset

from textMG.main_multiGPU import do_train, do_eval, Predict
from textMG.tf_serving.grpc_infer import Inference
from textMG.tf_serving.producer import Producer
from textMG.APIs.routers.users_token import get_current_user

from time import time
import json

dataset = Dataset()

template = Jinja2Templates("/home/projects/TFpackageText/textMG/APIs/htmls")
router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
    dependencies=[Depends(get_current_user)]
)
def initializer():
    #start producer
    producer = Producer()
    return producer

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
    ts=time()
    logger.debug("inference api is called, the inference is started")
    logger.debug("batch_infer is: {}".format(req))
    print("batch_infer is: {}".format(req))
    input = dataset.inquiry_process_pred(req.batch_infer)
    logger.info("time_used for data processing: {:.4f}".format(time()-ts))
    infer = Inference.doInfer(input, args.server)
    logger.debug(infer.preds)
    print("******" * 15)
    return Response(data=json.dumps(infer.preds), response_code=200, message="success", error=False)

@router.post("/pro_cons_infer")
def producer_consumer_infer(req: InferModel):
    """
    call producer_consumer_infer function
    please start Consumer component first prior to this API
    """
    logger.debug("iproducer_consumer_infer is called, the inference is started")
    logger.debug("batch_infer is: {}".format(req))
    print("batch_infer is: {}".format(req))
    #push query into a queue
    producer = initializer()
    response = producer.call(req.batch_infer)
    logger.debug(response)
    print("******" * 15)
    return Response(data=response, response_code=200, message="success", error=False)

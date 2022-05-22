#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/20 13:13
"""
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import sys
import threading
import grpc
from grpc import RpcError
import numpy as np
from time import time
import tensorflow as tf
from textMG.configs.config import args
from textMG.datasets.dataset import Dataset
from textMG.APIs.api_loggers.api_logger import logger

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class Inference:
    def __init__(self):
        self.result = {}

    @classmethod
    def doInfer(cls, input, server):
        if len(input) < 1: raise "the input shape[0] should be more then 0"
        if not server.split(':')[-1] == '8500': raise "the serving port should be 8500"
        logger.debug("server is: {}".format(server))
        infer = cls()
        infer(input, server)
        return infer

    def inferFunc(self, input, server):
        ts = time()
        with grpc.insecure_channel(server) as channel:
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

            # create infer request
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'multi_cnn_category_tf1_serving'
            # request.model_spec.signature_name = 'serving_default'
            request.inputs['input'].CopyFrom(
                tf.compat.v1.make_tensor_proto(input, dtype=float))

            # prediction
            logger.info("Attempting to predict against TF Serving API.")
            try:
                output = stub.Predict.future(request)  # 10 secs timeout
            except RpcError as err:
                logger.critical("stub.Predict.future error occurred: {}".format(err))
                raise

            model_version = output.result().model_spec.version.value
            logger.info("model_version used is: {}".format(model_version))
            logits = output.result().outputs['logits_prob'].float_val
            preds = output.result().outputs['prediction'].int64_val

            self.result['logits'] = str(logits)
            self.result['preds'] = str(preds)
            self.result['infer_time'] = '{:.4f}'.format(time()-ts)
            logger.info('time_used for infer: {:.4f}'.format(time()-ts))

    def __call__(self, input, server, *args, **kwargs):
        return self.inferFunc(input, server)

    @property
    def preds(self):
        return self.result


if __name__ == '__main__':
    # load the test data from dataset
    t1 = time()
    dataset = Dataset()
    input, _ = dataset.process_data(args.path_data_dir, args.vocab_file, args.path_stopwords,
                                    n_examples=args.num_tests)
    logger.info("timed used for data loading :{:.4f}s".format(time() - t1))

    t2 = time()
    infer = Inference.doInfer(input, args.server)
    result = infer.preds
    logger.info('num_tests: {}'.format(args.num_tests))
    if args.print_outputs:
        logger.info(result)
    logger.info("time_used: {:.4f}s".format(time() - t2))

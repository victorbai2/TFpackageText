#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/22 10:34
"""
import pika
import base64
import os, sys
import json
import numpy as np
from textMG.tf_serving.grpc_infer import Inference
from textMG.configs.config import args
from textMG.utils.loggers import logger


class Consumer(object):
    def __init__(self):
        self.queue_name = 'msg_queue'
        self.config = pika.ConnectionParameters(
            host='127.0.0.1',
            credentials=pika.PlainCredentials('victor', 'Nicaine!'),
        )
        self.connection = pika.BlockingConnection(self.config)
        self.channel = self.connection.channel()
        self.q = self.channel.queue_declare(queue=self.queue_name)

    def on_request(self, ch, method, props, body):
        logger.info('consumer-->received message：%s' % body)
        input = json.loads(body.decode("utf-8"))['input']
        infer = Inference.doInfer(input, args.server)
        logger.debug("inference result: {}".format(infer.preds))
        response_data = json.dumps(infer.preds)
        logger.info("queue_size: {}".format(self.q.method.message_count))
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id= \
                                                             props.correlation_id),
                         body=response_data)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("******" * 15)
    def on_consume(self):
        self.channel.basic_qos(prefetch_count=10)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.on_request)

        logger.info("Awaiting RPC requests.....")
        self.channel.start_consuming()


if __name__ == "__main__":
    consumer = Consumer()
    consumer.on_consume()
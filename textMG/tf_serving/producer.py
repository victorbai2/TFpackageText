#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/22 10:34
"""
import pika
import uuid
import base64
import json
import time
from textMG.datasets.dataset import Dataset
from textMG.utils.loggers import logger
dataset = Dataset()

class Producer(object):
    def __init__(self):
        self.queue_name='msg_queue'
        self.config = pika.ConnectionParameters(
            host='127.0.0.1',
            credentials=pika.PlainCredentials('victor', 'Nicaine!'),
        )
        self.connection = pika.BlockingConnection(self.config)
        self.channel = self.connection.channel()
        result = self.channel.queue_declare('', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(queue=self.callback_queue, on_message_callback=self.on_response, auto_ack=False)

    def pic_base64(self,filename):
        byte_content = open(filename, 'rb').read()
        ls_f = base64.b64encode(byte_content)
        return ls_f

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, message):
        ts = time.perf_counter()
        input_processed = dataset.inquiry_process_pred(message)
        json_values = {'input': input_processed}
        body_str=json.dumps(json_values)
        logger.info("time_used for data processing: {:.4f}".format(time.perf_counter() - ts))

        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.queue_name,
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=body_str)

        while self.response is None:
            self.connection.process_data_events()
        return self.response.decode("utf-8")

if __name__=='__main__':
    producer = Producer()
    for i in range(2):
        response = producer.call(["我爱北京天安门", "我爱北京天安门"])
        print("response message: {}".format(response))
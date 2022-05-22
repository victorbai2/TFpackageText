#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/11 18:33
"""

import tensorflow as tf
from textMG.models.bertBaseModule import BertModel
from textMG.models.lstm_crf_layer import BLSTM_CRF


class Bert_module:

    def __init__(self, *args, **kwargs):
        self.bert = BertModel
        self.kwargs = kwargs

    def __call__(self, num_labels, max_len, hidden_size, reuse=tf.AUTO_REUSE, is_training=True, dropout=0.3, *args, **kwargs):
        with tf.variable_scope('bert_classification', reuse=reuse):
            output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

            input_ids = tf.reshape(kwargs['input_ids'], [-1, max_len], name="vic_input_ids")
            input_mask = tf.reshape(kwargs['input_mask'], [-1, max_len], name="vic_input_mask")
            input_type_ids = tf.reshape(kwargs['input_type_ids'], [-1, max_len], name="vic_input_type_ids")

            embeddings_pretrained = self.bert(
                config=kwargs['bert_config'],
                input_ids=tf.cast(input_ids, dtype=tf.int32),
                input_mask=tf.cast(input_mask, dtype=tf.int32),
                token_type_ids=tf.cast(input_type_ids, dtype=tf.int32),
                is_training=kwargs['is_training_pretrained'],
                use_one_hot_embeddings=kwargs['use_one_hot_embeddings'])

            #retrieve last layer
            embeddings = embeddings_pretrained.get_pooled_output()
            dropout_1 = tf.layers.dropout(embeddings, rate=dropout, training=is_training, name="dropout_1")

            output = tf.nn.bias_add(tf.matmul(dropout_1, output_weights, transpose_b=True), output_bias)
            output = tf.nn.softmax(output, name='logits_prob') if not is_training else output

            result = {"output_layer": embeddings, "dropout_1":dropout_1, "output": output}

            return result

class Bert_lstm_crf:

    def __init__(self, *args, **kwargs):
        self.bert = BertModel
        self.kwargs = kwargs

    def __call__(self, num_labels, max_len, hidden_size, reuse=tf.AUTO_REUSE, is_training=True, *args, **kwargs):
        with tf.variable_scope('bert_classification', reuse=reuse):
            input_ids = tf.reshape(kwargs['input_ids'], [-1, max_len], name="vic_input_ids")
            input_mask = tf.reshape(kwargs['input_mask'], [-1, max_len], name="vic_input_mask")
            input_type_ids = tf.reshape(kwargs['input_type_ids'], [-1, max_len], name="vic_input_type_ids")

            embeddings_pretrained = self.bert(
                config=kwargs['bert_config'],
                input_ids=tf.cast(input_ids, dtype=tf.int32),
                input_mask=tf.cast(input_mask, dtype=tf.int32),
                token_type_ids=tf.cast(input_type_ids, dtype=tf.int32),
                is_training=kwargs['is_training_pretrained'],
                use_one_hot_embeddings=kwargs['use_one_hot_embeddings'])

            #retrieve last layer
            embeddings = embeddings_pretrained.get_sequence_output()
            max_seq_length = embeddings.shape[1].value
            #get the actual length of sequence
            used = tf.sign(tf.abs(input_ids))
            lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size]
            # add CRF output layer
            blstm_crf = BLSTM_CRF(embedded_chars=embeddings, hidden_unit=hidden_size, cell_type=self.kwargs.cell,
                                  num_layers=self.kwargs.num_layers, dropout_rate=self.kwargs.dropout_rate,
                                  initializers=self.kwargs.initializers, num_labels=num_labels,
                                  seq_length=max_seq_length, labels=self.kwargs.labels, lengths=lengths,
                                  is_training=is_training)
            rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
            return rst
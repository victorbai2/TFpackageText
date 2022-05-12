#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/11 21:06
"""
import tensorflow as tf
from models.bert_module import Bert_module
from models.bertBaseModule import BertConfig, get_assignment_map_from_checkpoint

from textMG.datasets.generator import Generator
from textMG.datasets.dataset import Dataset
from textMG.utils.loss import loss_function
from textMG.configs.config import args


generator = Generator(is_pretrained=True)
# print 2 outputs from our generator just to see if it works:
iter = generator.get_next_patch(batch=2)
el = next(iter)
print('input_ids shape: {}'.format([len(el['input_ids']), len(el['input_ids'][0])]))
print('input_masks shape: {}'.format([len(el['input_masks']), len(el['input_masks'][0])]))
print('input_type_ids shape: {}'.format([len(el['input_type_ids']), len(el['input_type_ids'][0])]))
print('y_output shape: {}'.format([len(el['y_output']), len(el['y_output'][0])]))

input_ids = el['input_ids']
input_masks = el['input_masks']
input_type_ids = el['input_type_ids']
y_output = el['y_output']

# dataset = Dataset()
# input_ids, input_masks, input_type_ids, y_output = dataset.process_data_pretrained(args.path_data_dir, args.pretrained_vocab_file, args.path_stopwords, max_len=args.max_len, is_token_b=False, n_examples=3)

bert_config = BertConfig.from_json_file(args.bert_config_file)
input_params = {
    'bert_config': bert_config,
    'input_ids': input_ids,
    'input_mask': input_masks,
    'input_type_ids': input_type_ids,
    'is_training_pretrained': True,
    'use_one_hot_embeddings': False,
}


model = Bert_module()
result = model(args.num_classes, args.max_len, args.hidden_size, reuse=tf.AUTO_REUSE, is_training=True, **input_params)
pred = result['output']

if not isinstance(y_output, tf.Tensor):
    y = tf.cast(y_output, dtype=tf.float32)

loss = loss_function(pred, y)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accur = tf.reduce_mean(tf.cast(corr, tf.float32))

def load_init_from_checkpoint(init_checkpoint):
    # here to restore pretrained model
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(
        tvars, args.init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

if args.init_checkpoint:
    load_init_from_checkpoint(args.init_checkpoint)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1):
        outp1 = sess.run(pred)
        _, loss, accuracy = sess.run([train_op, loss, accur])
        outp2 = sess.run(pred)
        print(loss)
        print(accuracy)
    print("finished")


print('x_ids_batchG.shape: {}, x_masks_batchG.shape: {}, x_type_ids_batchG.shape:'
                                        ' {}, x_batchG.shape: {}'.format(1,2,3,4))
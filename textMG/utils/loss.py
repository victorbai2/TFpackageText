import tensorflow as tf
from textMG.utils.loggers import logger
from typing import Union, List


def loss_function(pred: tf.Tensor, y: Union[tf.Tensor, List]) -> tf.Tensor:
    if not isinstance(pred, tf.Tensor):
        pred = y = tf.cast(pred, dtype=tf.float32)
    if not isinstance(y, tf.Tensor):
        y = tf.cast(y, dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    return loss


def weighted_loss(pred: tf.Tensor, y: Union[tf.Tensor, List], weights: List[Union[int, float]],
                  use_predefined_loss: bool = True) -> tf.Tensor:
    if not isinstance(pred, tf.Tensor):
        pred = y = tf.cast(pred, dtype=tf.float32)
    weights = tf.cast(weights, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    try:
        weights.get_shape() == y.get_shape()[-1]
    except Exception as e:
        logger.critical("RuntimeError occurred", exc_info=1)
        logger.critical("please use the accurate shape of weights as label: ".format(y.get_shape()))
        raise e
    if not use_predefined_loss:
        logits_softmax_log = tf.math.log(tf.nn.softmax(pred))
        loss = -tf.math.multiply(y, logits_softmax_log)
        weighted_loss_ = tf.math.multiply(loss, weights)
        weighted_loss = tf.reduce_mean(tf.reduce_sum(weighted_loss_, axis=-1))
    else:
        y_weighted = tf.math.multiply(y, weights)
        weighted_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_weighted))

    return weighted_loss
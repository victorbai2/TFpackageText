import tensorflow as tf
from typing import Optional, Union, List


def accuracy_cal(pred: tf.Tensor, y: Union[tf.Tensor, List]) -> tf.Tensor:
    if not isinstance(y, tf.Tensor):
        y = tf.cast(y, dtype=tf.float32)
    corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accur = tf.reduce_mean(tf.cast(corr, tf.float32))
    return accur
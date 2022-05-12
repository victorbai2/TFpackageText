import tensorflow as tf

def accuracy_cal(pred, y):
    if not isinstance(y, tf.Tensor):
        y = tf.cast(y, dtype=tf.float32)
    corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accur = tf.reduce_mean(tf.cast(corr, tf.float32))
    return accur
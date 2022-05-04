import tensorflow as tf

def accuracy_cal(pred, y):
    corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accur = tf.reduce_mean(tf.cast(corr, tf.float32))
    return accur
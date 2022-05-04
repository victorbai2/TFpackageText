import tensorflow as tf

def loss_function(pred, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    return loss
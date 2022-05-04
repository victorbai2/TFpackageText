# Build a convolutional neural network
import tensorflow as tf
from textMG.configs.config import args
from textMG.embeddings.word_embeddings import Get_embeddings
from textMG.datasets.generator import Generator

class Model_cnn:
    def __init__(self):
        self.embeddings = Get_embeddings().get_embeddings()

    def placeholder_init(self, num_input, num_classes):
        X = tf.placeholder(tf.float32, [None, num_input], name='x_input')
        Y = tf.placeholder(tf.float32, [None, num_classes], name='y_output')
        return X, Y

    def __call__(self, input_x, num_classes, dropout, reuse, is_training, *args, **kwargs):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            embeddings = tf.get_variable(name="w_embedding", shape=self.embeddings.shape,
                                        initializer=tf.constant_initializer(self.embeddings),
                                        trainable=True)
            input_x = tf.cast(input_x, dtype=tf.int32, name='input_x')
            # input_x = tf.constant(input_x, dtype=tf.int32, name='input_x')
            embedding_input = tf.nn.embedding_lookup(embeddings, input_x)

            input_x_shaped = tf.reshape(embedding_input, shape=[-1, 30, 50, 1], name='reshape_embeddings')
            # Convolution Layer with 64 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(input_x_shaped, 64, 5, activation=tf.nn.relu, name='conv1')
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')

            # Convolution Layer with 256 filters and a kernel size of 5
            conv2_1 = tf.layers.conv2d(pool1, 256, 3, activation=tf.nn.relu, name='conv2_1')
            # Convolution Layer with 512 filters and a kernel size of 5
            conv2_2 = tf.layers.conv2d(conv2_1, 512, 3, activation=tf.nn.relu, name='conv2_2')
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, name='pool2')

            # Flatten the data to a 1-D vector for the fully connected layer
            flatten = tf.contrib.layers.flatten(pool2)

            # Fully connected layer (in contrib folder for now)
            fc1 = tf.layers.dense(flatten, 2048, name='fc1')
            # Apply Dropout (if is_training is False, dropout is not applied)
            drop1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

            # Fully connected layer (in contrib folder for now)
            fc2 = tf.layers.dense(drop1, 1024, name='fc2')
            # Apply Dropout (if is_training is False, dropout is not applied)
            drop2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)

            # Output layer, class prediction
            output = tf.layers.dense(drop2, num_classes, name='logits')
            # Because 'softmax_cross_entropy_with_logits' loss already apply
            # softmax, we only apply softmax to testing network
            output = tf.nn.softmax(output, name='logits_prob') if not is_training else output

            # RETURN
            result = {'input': input_x, 'embedding_input': embedding_input, 'conv1': conv1, 'pool1': pool1,
                      'conv2_1': conv2_1,
                      'conv2_2': conv2_2, 'pool2': pool2, 'flatten': flatten,
                      'fc1': fc1, 'drop1': drop1, 'fc2': fc2, 'drop2': drop2, 'output': output
                      }
            return result


if __name__=="__main__":
    model = Model_cnn()
    generator = Generator()
    iter = generator.get_next_patch(batch=1)
    input=next(iter)
    pred = model(input['x_input'], args.num_classes, args.dropout, reuse=tf.AUTO_REUSE, is_training=False)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(pred))
    sess.close()
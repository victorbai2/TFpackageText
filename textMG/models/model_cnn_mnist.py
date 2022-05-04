# Build a convolutional neural network
import tensorflow as tf

def placeholder_init(num_input, num_classes):
    X = tf.placeholder(tf.float32, [None, num_input], name='x_input')
    Y = tf.placeholder(tf.float32, [None, num_classes], name='y_output')
    return X, Y

def conv_net(input_x, num_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        input_x_shaped = tf.reshape(input_x, shape=[-1, 28, 28, 1])

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
        result = {'input': input_x, 'input_x_shaped': input_x_shaped, 'conv1': conv1, 'pool1': pool1,
                  'conv2_1': conv2_1,
                  'conv2_2': conv2_2, 'pool2': pool2, 'flatten': flatten,
                  'fc1': fc1, 'drop1': drop1, 'fc2': fc2, 'drop2': drop2, 'output': output
                  }
        return result
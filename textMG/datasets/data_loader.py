from tensorflow.examples.tutorials.mnist import input_data

def load_data(data_load_dir):
    mnist = input_data.read_data_sets(data_load_dir, one_hot=True)
    return mnist
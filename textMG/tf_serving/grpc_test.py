from __future__ import print_function

# This is a placeholder for a Google-internal import.
import numpy as np
import grpc
from time import time
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_integer('num_tests', 1, 'Number of test images')
FLAGS = tf.compat.v1.app.flags.FLAGS

def main(num_tests):
  t1 = time()
  result = []
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  for _ in range(num_tests):
    data = np.random.normal(loc=3, scale=2.5, size=(1, 784))
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "multi_cnn_mnist_tf1_serving"
    # request.model_spec.signature_name = 'serving_default'
    request.inputs['input'].CopyFrom(
        tf.compat.v1.make_tensor_proto(data, dtype=float))
    # output = stub.Predict(request, 10.0)  # 10 secs timeout
    output = stub.Predict.future(request, 10.0) # 10 secs timeout
    model_version = output.result().model_spec.version.value
    print('model_version :', model_version)
    response = output.result().outputs['prediction'].int64_val
    result.append(response)
  t2 = time()
  print(result)
  print("time_used: {:.4f}s".format(t2-t1))

if __name__ == '__main__':
  main(FLAGS.num_tests)
  # tf.compat.v1.app.run()

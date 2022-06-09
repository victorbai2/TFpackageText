import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import requests
import numpy as np
import json
from time import time
from textMG.configs.config import args
from textMG.datasets.dataset import Dataset
import argparse
from typing import List

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--model_version', type=str, default="001", help="model version to be used for prediction")
parser.add_argument('--num_tests', type=int, default=500, help="number inputs to be used for prediction")
parser.add_argument('--print_outputs', type=bool, default=False, help="whether or not to print output predictions")
FLAGS = parser.parse_args()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def pred_func(input: List[int], num_tests: int, server_curl: str) -> None:
    t1 = time()
    results = []
    # for i in range(num_tests):
    param = json.dumps({"instances": input}, cls=NumpyEncoder)
    response = requests.post(server_curl, data=param)
    response.raise_for_status()
    # print("response code:",  response.status_code)
    #load the response with json
    response_dict = json.loads(response.text)
    #retore the result
    # prediction = response_dict['predictions'][0]['prediction']
    # res = [key for key, value in label_dict.items() if value == prediction]
    # results.append(res)

    # for d in response_dict['predictions']:
    # 	results.append(d['prediction'])
    t2 = time()
    if FLAGS.print_outputs:
        print(response_dict)
    print("time_used: {:.4f}s".format(t2-t1))
if __name__ == "__main__":
    # load the test data from dataset
    t1 = time()
    dataset = Dataset()
    input, _ = dataset.process_data(args.path_data_dir, args.vocab_file, args.path_stopwords, n_examples=FLAGS.num_tests)
    print("timed used for data loading :{:.4f}s".format(time()-t1))
    server_curl = 'http://192.168.1.5:8501/v1/models/multi_cnn_category_tf1_serving/versions/' + FLAGS.model_version + ':predict'
    print('model_version :', FLAGS.model_version)
    print('num_tests :', FLAGS.num_tests)
    pred_func(input, FLAGS.num_tests, server_curl)

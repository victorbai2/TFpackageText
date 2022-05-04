# TFpackageText

<details open="open">
<summary>Table of Contents</summary>

- [Prerequisites](#Prerequisites)
- [Getting Started](#Getting-Started)
  - [Installation](#Installation)
  - [Usage](#Usage)
  - [Model Training/evaluating/prediction](#Model-Training/evaluating/prediction)
  - [API calls](#API-calls)
  - [Swagger docs](#Swagger-docs)
  - [Data Pipeline](#Data-Pipeline)
  - [Tensorflow serving](#Tensorflow-serving)
  - [Examples](#Examples)
  - [pressure test](#pressure-test)
- [Contributing](#Contributing)
- [License](#License)
</details>

---

## Prerequisites
* python >= 3.7
* tensorflow == 1.15
* fastapi == 0.75.2

## Getting Started

### Installation
```
$ pip install -r requirements
```
or 
```
python setup.py sdist
pip install TFpackageText-0.0.1.tar.gz
```

### Usage
```
from textMG import models, datasets, APIs
```

### Model Training/evaluating/prediction
It is initially trained on multi-GPU environment
```
$ ./starup.sh -m [train|evel|pred]
```
or via APIs:
```
http://localhost:5000/api/train
http://localhost:5000/api/eval
http://localhost:5000/api/pred
```

### API calls
Start API
```
$ python api_run.py
```

### Swagger docs
```
http://localhost:5000/docs
```
### Data Pipeline
data can be loaded directly
```python
# load data from disk and generator.
generator = Generator()
# print 2 outputs from our generator just to see if it works:
iter = generator.get_next_patch(batch=2)
for i in range(2):
    el = next(iter)
```
or via tf.data.Dataset.from_generator
```python
train_dataset = tf.data.Dataset.from_generator(generator.get_next_patch,
                                               output_types={generator.input: tf.float32,
                                                             generator.output: tf.float32})
train_dataset = train_dataset.repeat().batch(args.batch_size * args.num_gpusORcpus).prefetch(1)
```
or via api
```python
curl -X 'GET' \
  'http://localhost:5000/api/batch_load/2' \
  -H 'accept: application/json'
```
### Tensorflow serving
after training, the model for tf servering is saved to different path for restful and grpc.
```python
# call build_SavedModel to build a SavedModel for tensor serving
build_SavedModel(conv_net, args.export_path_serving, args.savedmodel_version, X_batch_numGPU, best_sess)
```

### Examples
predict:
```
curl -X 'POST' \
  'http://localhost:5000/api/pred' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inquiry": [
    "我热爱...", "喜欢游泳...."...
  ]
}'

Response Body:

{
  "data": {
    "records": {
      "0": {
        "category": "entertainment",
        "label": "1",
        "pred_logits": "[0.17890048027038574, 0.23429206013679504, 0.220640629529953, 0.17155364155769348, 0.1946132332086563]"
      },
      "1": {
        "category": "entertainment",
        "label": "1",
        "pred_logits": "[0.17201262712478638, 0.24676887691020966, 0.22940053045749664, 0.15629342198371887, 0.19552461802959442]"
      }
    },
    "time_used": "2.519162654876709"
  },
  "code": 200,
  "message": "success",
  "error": false
}
```
load data batch(batch_size=2):
```
curl -X 'GET' \
  'http://localhost:5000/api/batch_load/2' \
  -H 'accept: application/json'
  
Response Body:
{
  "data": {
    "0": {
      "x_input": [
        [701,101,..],
        [1842,2182,...]
      ],
      "y_output": [
        [0,0,0,0,1],
        [0,0,0,0,1],
      ]
    },
    "1": {
      "x_input": [
        [3934,150,...],
        [3181,5275,...]
      ],
      "y_output": [
        [0,0,0,0,1],
        [0,0,0,0,1],
      ]
    },
    "input_shape": [2,30],
    "output_shape": 2
  },
  "code": 200,
  "message": "success",
  "error": false
}
```

### pressure test
```
$ab -p 'tensorAPI.json'  -T 'application/json' -H 'Content-Type: application/json'  -c 500 -n 500 -t 120 'http://192.168.1.14:5000/api/pred'
```

tensorAPI.json
```
{
  "inquiry": [
    "sentence1", "sentence2" ...
  ]
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[VB](https://VB.com/licenses.cn)
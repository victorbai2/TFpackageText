import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--mode', type=str, help="Mode: 'train' or 'eval' or 'pred'")
# parser.add_argument('--status',choices=['train','test'],default='train')
parser.add_argument('--device_type', type=str, default="CPU", help="Device: 'CPU' or 'GPU'")
parser.add_argument('--num_gpusORcpus', type=int, default=2)
parser.add_argument('--log_device_placement', type=bool, default=False,
                    help="Whether device placements should be logged")
parser.add_argument('--num_input', type=int, default=784)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--max_len', type=int, default=30)
parser.add_argument('--dim_size', type=int, default=50)
parser.add_argument('--total_examples', type=int, default=90000)
parser.add_argument('--model_dir', type=str, default="/home/projects/tensorflow_practice/save_model/multi_cnn_category_tf1/")
parser.add_argument('--export_path_serving', type=str, default="/home/projects/tensorflow_practice/save_model/multi_cnn_category_tf1/model_serving")
parser.add_argument('--savedmodel_version', type=str, default="001",
                    help="version number must be a string with 3 digitals")
parser.add_argument('--path_rawdata', type=str, default="/home/projects/tensorflow_practice/data_classification/raw_data/")
parser.add_argument('--path_data_dir', type=str, default="/home/projects/tensorflow_practice/data_classification/data/")
parser.add_argument('--vocab_file', type=str, default="/home/projects/tensorflow_practice/data_classification/vocab/")
parser.add_argument('--path_stopwords', type=str, default="/home/projects/tensorflow_practice/data_classification/stopwords/stopwords.txt")
parser.add_argument('--data_load_dir', type=str, default="/home/projects/tensorflow_practice/tensorflow_practice/data")
parser.add_argument('--path_word_embeddings', type=str, default="/home/projects/tensorflow_practice/data_classification/word2vec_model/")
# parser.add_argument('--dumpedd_data_dir', type=str, default="../../../dumped_data")
label_dict = {'car':0, 'entertainment':1, 'military':2, 'sports':3, 'technology':4}
parser.add_argument('--training_epochs', type=int, default=3)
parser.add_argument('--learning_rate', type=int, default=0.001)
parser.add_argument('--dropout', type=int, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--weights_for_loss', type=list, default=[1,1,1,1,1])
#tensorflow serving
parser.add_argument('--concurrency', type=int, default=10, help='maximum number of concurrent inference requests')
parser.add_argument('--num_tests', type=int, default=1, help='Number of test images')
parser.add_argument('--server', type=str, default='192.168.1.5:8500', help='PredictionService host:port')
parser.add_argument('--print_outputs', type=bool, default=False, help='whether or not to print output predictions')
#bert related args
parser.add_argument('--is_pretrained', type=bool, default=False)
parser.add_argument('--pretrained_vocab_file', type=str, default="/home/projects/tensorflow_practice/data_classification/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt")
parser.add_argument('--do_lower_case', type=bool, default=True)
parser.add_argument('--bert_config_file', type=str, default="/home/projects/tensorflow_practice/data_classification/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json")
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--init_checkpoint', type=str, default="/home/projects/tensorflow_practice/data_classification/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt")
parser.add_argument('--model_dir_save_pretrain', type=str, default="/home/projects/tensorflow_practice/save_model/multi_bert_category_tf1/")
#api args
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--host', type=str, default='192.168.1.14')
args, remaining_argv = parser.parse_known_args()


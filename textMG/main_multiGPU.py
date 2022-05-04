from __future__ import division, print_function, absolute_import
import warnings

from utils.loggers import logger

warnings.filterwarnings('ignore')
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
# print("tensorflow version:", tf.__version__)
# tf.compat.v1.logging.info("tensorflow version: {}".format(tf.__version__))
logger.info("tensorflow version: {}".format(tf.__version__))

import numpy as np
from collections import deque
from time import time
from sklearn.metrics import f1_score
import argparse

from configs.config import parser, args, remaining_argv, label_dict
from models.model_cnn_category import Model_cnn
from datasets.dataset import Dataset
from tf_serving.savedmodel_serving import build_SavedModel
from configs.config_multiGPU import multi_GPU_training, device_set
from utils import metrics, pred_to_result, json_output


def loader_before_training():
    # load evaluation data
    dataset = Dataset()
    x_eval, y_eval = dataset.process_data(args.path_data_dir, args.vocab_file, args.path_stopwords, n_examples=500)
    # load model
    conv_net = Model_cnn()
    # train on CPU or GPU
    device_set(args.device_type)
    return dataset, conv_net, x_eval, y_eval


# do_train
def do_train():
    start = time()
    # Place all ops on CPU by default
    with tf.device('/cpu:0'):
        # call loader before start training or evaluating
        _, conv_net, x_eval, y_eval = loader_before_training()

        # do multiGPU training
        train_op, loss_patch_ave, train_initializer, X_batch_numGPU, Y_batch_numGPU = multi_GPU_training(conv_net)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # set the device configuration
        if args.device_type == "CPU":
            config = tf.ConfigProto(intra_op_parallelism_threads=args.num_gpusORcpus,
                                    inter_op_parallelism_threads=args.num_gpusORcpus,
                                    allow_soft_placement=True, log_device_placement=args.log_device_placement,
                                    device_count={'CPU': args.num_gpusORcpus})
        elif args.device_type == "GPU":
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_placement,
                                    device_count={'GPU': args.num_gpusORcpus})
        # create the graph
        with tf.Session(config=config) as sess:
            sess.run(init)
            sess.run(train_initializer)
            # saver
            saver = tf.train.Saver(max_to_keep=2)
            # early stopping and save the best
            test_ac_history = deque(maxlen=args.patience + 1)
            max_test_acc = 0
            best_epoch = args.training_epochs
            best_sess = None
            is_saved = None
            total_batch = int(args.total_examples / (args.batch_size * args.num_gpusORcpus))
            # Keep training until reach max iterations
            for epoch in range(1, args.training_epochs + 1):
                ts = time()
                avg_train_cost = 0
                for i in range(total_batch):
                    # Get a batch from tf.data.Dataset.from_generator to verify the batch.
                    if epoch == 1 and i == 0:  # only print for the 1th epoch and 1st patch
                        x_batch, y_batch = sess.run([X_batch_numGPU, Y_batch_numGPU])
                        logger.info('x_batch.shape: {}, y_batch.shape: {}'.format(x_batch.shape, y_batch.shape))

                    # Run optimization op (backprop)
                    _, train_cost = sess.run([train_op, loss_patch_ave])
                    # Compute average loss
                    avg_train_cost += train_cost / total_batch
                    if i >= 3:
                        break
                # get the accuracy for train and test dataset
                x_train = x_eval[:250]
                y_train = y_eval[:250]
                pred_tr = conv_net(x_train, args.num_classes, args.dropout,
                                   reuse=True, is_training=False)['output']
                train_acc_ = metrics.accuracy_cal(pred_tr, y_train)
                train_acc = sess.run(train_acc_)
                # test accuracy
                x_test = x_eval[250:]
                y_test = y_eval[250:]
                pred_te = conv_net(x_test, args.num_classes, args.dropout,
                                   reuse=True, is_training=False)['output']
                test_acc_ = metrics.accuracy_cal(pred_te, y_test)
                test_acc = sess.run(test_acc_)

                # save best net
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    best_sess = sess
                    best_epoch = epoch
                # early stopping
                test_ac_history.append(test_acc)
                if len(test_ac_history) > args.patience:
                    if test_ac_history.popleft() > max(test_ac_history):
                        logger.info('Early stopping. No loss improvement, stopped at epoch {}.'.format(epoch))
                        # best_epoch = epoch - args.patience - 1
                        logger.info('best_epoch is : {}'.format(best_epoch))
                        # save model for both eval/pred
                        saver.save(best_sess, args.model_dir + "cnn_category.ckpt-" + str(best_epoch))
                        is_saved = 1
                        logger.info("saved model to: {}".format(args.model_dir))
                        # call build_SavedModel to build a SavedModel for tensor serving
                        build_SavedModel(conv_net, args.export_path_serving, args.savedmodel_version, X_batch_numGPU,
                                         best_sess)
                        break

                # get time used for each epoch
                time_used = time() - ts
                # print
                if epoch % args.display_step == 0:
                    # F1_score for train
                    train_macro = f1_score(np.argmax(y_train, -1), np.argmax(pred_tr.eval(session=sess), -1),
                                           average='macro')
                    # F1_score for test
                    test_macro = f1_score(np.argmax(y_test, -1), np.argmax(pred_te.eval(session=sess), -1),
                                          average='macro')

                    logger.info(
                        "Epoch: {:02d}/{:02d}, avg_train_cost: {:.5f}, train_acc: {:.5f}, test_acc: {:.5f}, time_used: {:.2f}s"
                            .format(epoch, args.training_epochs, avg_train_cost, train_acc, test_acc, time_used))
                    logger.info("f1_score: train_macro:{:.4f}, test_macro:{:.4f}"
                                .format(train_macro, test_macro))
                    train_result = {
                        "epoches": str(epoch),
                        "avg_train_cost": str(avg_train_cost),
                        "train_acc": str(train_acc),
                        "test_acc": str(test_acc),
                        "train_macro": str(train_macro),
                        "test_macro": str(test_macro),
                        "saved model to path": args.model_dir
                    }
            # save model for both eval/pred and model serving
            if is_saved == None:
                saver.save(best_sess, args.model_dir + "cnn_category.ckpt-" + str(best_epoch))
                logger.info("saved model to: {}".format(args.model_dir))
                # call build_SavedModel to build a SavedModel for tensor serving
                build_SavedModel(conv_net, args.export_path_serving, args.savedmodel_version, X_batch_numGPU, best_sess)

    end = time()
    logger.info('Elapsed time is {:.2f} seconds.'.format(end - start))
    logger.info("Finished!")
    train_result["time_used"] = str(end - start)
    logger.debug(train_result)
    return train_result


# do_eval
def do_eval():
    with tf.device('/cpu:0'):
        # call loader before start training or evaluating
        _, conv_net, x_eval, y_eval = loader_before_training()

        with tf.Graph().as_default() as g:
            t1 = time()
            # test accuracy and F1_score
            x_test = x_eval
            y_test = y_eval
            pred_te = conv_net(x_test, args.num_classes, args.dropout, reuse=False, is_training=False)['output']
            test_acc_ = metrics.accuracy_cal(pred_te, y_test)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            sess.run(init)
            # restore model
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                logger.info("restoring mode from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, "save/cnn_mnist.ckpt-" + str(best_epoch))
            test_acc = sess.run(test_acc_)
            test_macro = f1_score(np.argmax(y_test, -1), np.argmax(pred_te.eval(session=sess), -1), average='macro')
            t2 = time()
            time_used = t2 - t1
            logger.debug("test_acc :{:.4f}, test_macro:{:.4f}, time_used: {:.2f}s"
                         .format(test_acc, test_macro, time_used))
            eval_result = {"restored model path": ckpt.model_checkpoint_path,
                           "test_accuracy": str(test_acc),
                           "test_macro": str(test_macro),
                           "time_used": str(time_used)
                           }
            logger.debug(eval_result)
            return eval_result


# do_pred
class Predict:
    def __init__(self):
        self.pred_result = None

    def do_pred(self, inquires):
        with tf.device('/cpu:0'):
            # load model
            conv_net = Model_cnn()
            with tf.Graph().as_default() as g:
                t1 = time()
                logger.info("input_length is: {}".format(len(inquires[0])))
                input = Dataset().inquiry_process_pred(inquires)
                pred_logit = conv_net(input, args.num_classes, args.dropout, reuse=False, is_training=False)['output']
                prediction = tf.argmax(pred_logit, -1)
                init = tf.global_variables_initializer()
                saver = tf.train.Saver()

            with tf.Session(graph=g) as sess:
                sess.run(init)
                # restore model
                ckpt = tf.train.get_checkpoint_state(args.model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    logger.info("restoring mode from: {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
                # saver.restore(sess, "save/cnn_mnist.ckpt-" + str(best_epoch))
                predictions, pred_logits = sess.run([prediction, pred_logit])
                categories = pred_to_result.predToResult(predictions, label_dict)
                t2 = time()
                time_used = t2 - t1
                logger.debug("categories: {}".format(categories))
                logger.debug("pred_logits: {}".format(pred_logits))
                logger.debug("time_used: {:.2f}s".format(time_used))
                self.pred_result = json_output.jsonOutput(categories, pred_logits.tolist(), label_dict, time_used)
                logger.debug(self.pred_result)

    def get_pred_result(self):
        return self.pred_result


if __name__ == '__main__':
    # choose mode
    if args.mode == 'train':
        logger.info("training is called")
        train_result = do_train()
    elif args.mode == 'eval':
        logger.info("evaluating is called")
        eval_result = do_eval()
    elif args.mode == 'pred':
        logger.info("prediction is called")
        inquires = []
        parser_parent = argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter, parents=[parser])
        # parser_parent.add_argument('--input_shape', nargs='+', type=int, required=True,
        #                            help="the input_shape requires two arguments, e.g. 'input_shape 1 784' ")
        args = parser_parent.parse_args(remaining_argv)
        inquiry = input("Please enter a sentence: ")
        # inquiry = "我想去旅行"
        inquires.append(inquiry)
        pre = Predict()
        pre.do_pred(inquires)
    else:
        logger.info(parser.print_help())
        logger.info("please specify the mode: 'train' or 'eval' or 'pred'")

import tensorflow as tf
import os
import shutil
from textMG.configs.config import args


def build_SavedModel(model_network, export_path_serving, savedmodel_version, X, best_sess):
    """
    @rtype: object
    """
    if isinstance(savedmodel_version, str):
        try:
            assert len(savedmodel_version) == 3
        except Exception as e:
            print('error: the version number {} must 3'.format(savedmodel_version))
            raise
    else:
        print('error: version must a string')
        raise
    model_path = os.path.join(export_path_serving, savedmodel_version)
    # if export_path_serving is existing, then remove
    if os.path.exists(model_path):
        print(savedmodel_version, "is an existing directory")
        try:
            shutil.rmtree(model_path)
            print(savedmodel_version, "is removed")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    # build a SavedModel for tensor serving
    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    model = model_network(X, args.num_classes, args.dropout,
                     reuse=True, is_training=False)
    prediction = tf.argmax(model['output'], -1)
    tensor_info_input = {'input': tf.saved_model.utils.build_tensor_info(X)}
    tensor_info_output = {
        'logits_prob': tf.saved_model.utils.build_tensor_info(model['output']),
        'prediction': tf.saved_model.utils.build_tensor_info(prediction),
        # 'conv1': tf.saved_model.utils.build_tensor_info(model['conv1']),
        # 'conv2_2': tf.saved_model.utils.build_tensor_info(model['conv2_2']),
    }
    method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs=tensor_info_input,
            outputs=tensor_info_output,
            method_name=method_name
        ))
    key_my_signature = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    builder.add_meta_graph_and_variables(
        best_sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={key_my_signature: prediction_signature},
        main_op=tf.tables_initializer(),
        strip_default_attrs=True,
        clear_devices=True)
    builder.save()
    print("saved model to:", model_path, "for serving")


def build_SavedModel_pretrained(model_network, export_path_serving, savedmodel_version, input_params, best_sess):
    if isinstance(savedmodel_version, str):
        try:
            assert len(savedmodel_version) == 3
        except Exception as e:
            print('error: the version number {} must 3'.format(savedmodel_version))
            raise
    else:
        print('error: version must a string')
        raise
    model_path = os.path.join(export_path_serving, savedmodel_version)

    model = model_network(args.num_classes, args.max_len, args.hidden_size, reuse=True,
                    is_training=False, **input_params)['output']

    builder = tf.saved_model.builder.SavedModelBuilder(model_path)
    prediction = tf.argmax(model['output'], -1)
    # 将输入张量与名称挂钩
    inputs = {
        'input_ids': tf.saved_model.utils.build_tensor_info(model.input_ids),
        'input_mask': tf.saved_model.utils.build_tensor_info(model.input_mask),
        'token_type_ids': tf.saved_model.utils.build_tensor_info(model.token_type_ids),
    }

    outputs = {
        'logits_prob': tf.saved_model.utils.build_tensor_info(model['output']),
        'prediction': tf.saved_model.utils.build_tensor_info(prediction),
    }

    #签名定义
    ner_signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        best_sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'ner_def':ner_signature_def
        }
    )
    builder.save()
    print("saved model to:", model_path, "for serving")

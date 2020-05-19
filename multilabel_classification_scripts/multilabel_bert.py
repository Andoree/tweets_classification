import codecs
import configparser
from datetime import datetime
import os

import pandas as pd
import tensorflow as tf
import numpy as np
import modeling
import optimization
import tokenization
from bert_preprocessing import create_examples, file_based_convert_examples_to_features, \
    convert_examples_to_features


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([5], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)  #### multi-label case
        labels = tf.cast(labels, tf.float32)
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        prediction = tf.cast(probabilities, tf.float32)
        threshold = float(0.5)
        prediction = tf.cast(tf.greater(prediction, threshold), tf.int64)
        acc, acc_op = tf.metrics.accuracy(label_ids, prediction)

        with tf.name_scope('summary'):
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('accuracy', acc)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            prediction = tf.cast(probabilities, tf.float32)
            threshold = float(0.5)
            prediction = tf.cast(tf.greater(prediction, threshold), tf.int64)
            acc, acc_op = tf.metrics.accuracy(label_ids, prediction)
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "accuracy": acc_op}, every_n_iter=10)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn,
                training_hooks=[logging_hook],
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            prediction = tf.cast(probabilities, tf.float32)
            threshold = float(0.5)
            prediction = tf.cast(tf.greater(prediction, threshold), tf.int64)
            acc, acc_op = tf.metrics.accuracy(label_ids, prediction)
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "accuracy": acc_op}, every_n_iter=2)
            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn,
                evaluation_hooks=[logging_hook]
            )
        else:
            print("mode:", mode, "probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder, num_labels):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, num_labels], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def train_and_evaluate(train_examples, eval_examples, max_seq_length, estimator, tokenizer, batch_size, eval_steps,
                       num_train_steps, output_dir, num_labels):
    train_path = os.path.join(output_dir, "train.tf_record")
    if not os.path.exists(train_path):
        open(train_path, 'w').close()
    eval_path = os.path.join(output_dir, "eval.tf_record")
    if not os.path.exists(eval_path):
        open(eval_path, 'w').close()

    file_based_convert_examples_to_features(
        train_examples, max_seq_length, tokenizer, train_path)
    file_based_convert_examples_to_features(
        eval_examples, max_seq_length, tokenizer, eval_path)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_path,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True)

    print(f'Beginning Training!')

    file_based_convert_examples_to_features(
        eval_examples, max_seq_length, tokenizer, eval_path)

    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_path,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=num_train_steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=None,
        start_delay_secs=0,
        throttle_secs=10,
    )
    current_time = datetime.now()
    tf.estimator.train_and_evaluate(
        estimator, train_spec, eval_spec
    )
    print("Training took time ", datetime.now() - current_time)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    eval_results_path = os.path.join(output_dir, "eval_results.txt")
    with tf.gfile.GFile(eval_results_path, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def create_output(predictions, num_labels):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    result_columns = [f"p_label_{label_id}" for label_id in range(1, num_labels + 1)]
    dff.columns = result_columns

    return dff


def get_estimator(model_fn, run_config, batch_size):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": batch_size})
    return estimator


def predict(test_df, estimator, tokenizer, max_seq_length, num_labels):
    test_df = test_df.reset_index(drop=True)
    predict_examples = create_examples(test_df, False, num_labels)

    test_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer)

    print('Beginning Predictions!')
    current_time = datetime.now()

    predict_input_fn = input_fn_builder(features=test_features, seq_length=max_seq_length, is_training=False,
                                        drop_remainder=False, num_labels=num_labels)
    predictions = estimator.predict(predict_input_fn)
    print("Prediction took time ", datetime.now() - current_time)
    output_df = create_output(predictions, num_labels=num_labels)
    return output_df

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

CLASSIFICATION_LABELS = ["EF", "INF", "ADR", "DI", "Finding"]
PSYTAR_TEXT_COLUMN = ["sentences"]


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
    #print("sequence output\n",model.get_sequence_output())
    #print("---")
    #print("Pooled output\n", model.get_pooled_output())
    #print("---")
    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()
    
    #print("Output layer", output_layer)
    hidden_size = output_layer.shape[-1].value
    #print("Hidden size", hidden_size)
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    print("Num labels", num_labels)
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            print("output_layer", output_layer)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        #print("Logits", logits)
        #print("Output_bias", output_bias)
        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)  #### multi-label case
        #print("Probabilities",probabilities) 
        labels = tf.cast(labels, tf.float32)
        #print("Labels", labels)
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)

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
                #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

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

                ## original eval metrics
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_loss": loss,
                # }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            prediction = tf.cast(probabilities, tf.float32)
            threshold = float(0.5)
            prediction = tf.cast(tf.greater(prediction, threshold), tf.int64)
            acc, acc_op = tf.metrics.accuracy(label_ids, prediction)
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss, "accuracy": acc_op}, every_n_iter=2)
            # accuracy = {"accuracy" : acc[1]}
            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn,
                evaluation_hooks=[logging_hook]
            )
            # output_spec = tf.estimator.EstimatorSpec(
            #    mode=mode,
            #    loss=total_loss,
            #    eval_metric_ops=eval_metrics,
            #    scaffold=scaffold_fn)
        else:
            print("mode:", mode, "probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
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
                tf.constant(all_label_ids, shape=[num_examples, len(CLASSIFICATION_LABELS)], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_output(predictions):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    result_columns = [f"p_{label}" for label in CLASSIFICATION_LABELS]
    dff.columns = result_columns

    return dff


def main():
    config = configparser.ConfigParser()
    config.read_file(codecs.open("config.ini", "r", "utf-8"))
    corpus_dir = config.get('INPUT', 'CORPUS_DIR')
    bert_vocab = config.get('INPUT', 'BERT_VOCAB')
    bert_init_chkpnt = config.get('INPUT', 'BERT_INIT_CHKPNT')
    bert_config = config.get('INPUT', 'BERT_CONFIG')
    batch_size = config.getint('PARAMETERS', 'BATCH_SIZE')
    num_train_epochs = config.getint('PARAMETERS', 'NUM_TRAIN_EPOCHS')
    warmup_proportion = config.getfloat('PARAMETERS', 'WARMUP_PROPORTION')
    max_seq_length = config.getint('PARAMETERS', 'MAX_SEQ_LENGTH')
    learning_rate = config.getfloat('PARAMETERS', 'LEARNING_RATE')
    save_checkpoints_steps = config.getint('PARAMETERS', 'SAVE_CHECKPOINTS_STEPS')
    save_summary_steps = config.getint('PARAMETERS', 'SAVE_SUMMARY_STEPS')
    output_dir = config.get('OUTPUT', 'OUTPUT_DIR')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classification_results_file = config.get('OUTPUT', 'RESULTS_FILE')

    train_df = pd.read_csv(os.path.join(corpus_dir, "train.csv"), encoding="utf-8")
    test_df = pd.read_csv(os.path.join(corpus_dir, "test.csv"), encoding="utf-8")
    dev_df = pd.read_csv(os.path.join(corpus_dir, "dev.csv"), encoding="utf-8")
    train_df[CLASSIFICATION_LABELS] = train_df[CLASSIFICATION_LABELS].astype(np.int32)
    test_df[CLASSIFICATION_LABELS] = test_df[CLASSIFICATION_LABELS].astype(np.int32)
    dev_df[CLASSIFICATION_LABELS] = dev_df[CLASSIFICATION_LABELS].astype(np.int32)
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=bert_vocab, do_lower_case=True)
    train_examples = create_examples(train_df)
    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_examples) / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    num_steps_in_epoch = int(len(train_examples) / batch_size * num_train_epochs) // num_train_epochs
    save_checkpoints_steps = num_steps_in_epoch
    train_path = os.path.join(output_dir, "train.tf_record")
    # Ztrain_path = os.path.join('./working', "train.tf_record")
    # filename = Path(train_file)
    if not os.path.exists(train_path):
        open(train_path, 'w').close()

    file_based_convert_examples_to_features(
        train_examples, max_seq_length, tokenizer, train_path)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_path,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=save_summary_steps,
        keep_checkpoint_max=1,
        save_checkpoints_steps=save_checkpoints_steps)
    print("BERT CONFIG", bert_config)
    bert_config = modeling.BertConfig.from_json_file(bert_config)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(CLASSIFICATION_LABELS),
        init_checkpoint=bert_init_chkpnt,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={"batch_size": batch_size})

    print(f'Beginning Training!')
    print("FN_INPUT", dir(train_input_fn))

    # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    eval_path = os.path.join(output_dir, "eval.tf_record")
    # eval_path = os.path.join('./working', "eval.tf_record")
    # filename = Path(train_file)
    if not os.path.exists(eval_path):
        open(eval_path, 'w').close()

    eval_examples = create_examples(dev_df)
    file_based_convert_examples_to_features(
        eval_examples, max_seq_length, tokenizer, eval_path)

    # This tells the estimator to run through the entire set.
    eval_steps = None

    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_path,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)

    # hooks=[train_logging_hook]

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=num_train_steps,
    )
    # hooks=[eval_logging_hook]
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
    # eval_results_path = os.path.join("./working", "eval_results.txt")
    with tf.gfile.GFile(eval_results_path, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    test_df = test_df.reset_index(drop=True)
    predict_examples = create_examples(test_df, False)

    test_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer)

    print('Beginning Predictions!')
    current_time = datetime.now()

    predict_input_fn = input_fn_builder(features=test_features, seq_length=max_seq_length, is_training=False,
                                        drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    print("Prediction took time ", datetime.now() - current_time)
    output_df = create_output(predictions)
    merged_df = pd.concat([test_df, output_df], axis=1)
    submission = merged_df.drop(['sentences'], axis=1)
    submission.to_csv(os.path.join(output_dir, classification_results_file), index=False)

    submission.head()


if __name__ == '__main__':
    main()

import codecs
import configparser
import os

import numpy as np
import pandas as pd
from bert_emdeddings_cnn import TextCNNWithDynamicEmbeddings
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping


def str_to_embedding(vector_str):
    for value in vector_str.split(','):
        try:
            float(value)
        except Exception:
            print(repr(value))
    vector_values = [float(value) for value in vector_str.split(',')]
    vector_values = np.array(vector_values)
    return vector_values


def main():
    config = configparser.ConfigParser()
    config.read('config_train_cnn_bert_emb.ini')
    tweet_tokens_embs_path = config.get('INPUT', 'TWEET_TOKENS_EMBS_PATH')
    test_start_line_id = config.getint('INPUT', 'TEST_START_LINE_ID')
    dev_start_line_id = config.getint('INPUT', 'DEV_START_LINE_ID')
    negative_examples_ratio = config.getint('INPUT', 'NEGATIVE_EXAMPLES_RATIO')
    labels_filepath = config.get('INPUT', 'LABELS_FILE')
    results_dir = config.get('OUTPUT', 'RESULTS_DIR')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    test_results_fname = config.get('OUTPUT', 'TEST_RESULTS_FNAME')
    dev_results_fname = config.get('OUTPUT', 'DEV_RESULTS_FNAME')
    test_results_path = os.path.join(results_dir, test_results_fname)
    dev_results_path = os.path.join(results_dir, dev_results_fname)
    embedding_size = config.getint('PARAMETERS', 'EMBEDDING_SIZE')
    max_tweet_length = config.getint('PARAMETERS', 'MAX_TWEET_LENGTH')
    test_size = 0.2
    dev_size = 0.1
    batch_size = config.getint('PARAMETERS', 'BATCH_SIZE')
    num_epochs = config.getint('PARAMETERS', 'NUM_EPOCHS')
    decision_threshold = config.getfloat('PARAMETERS', 'DECISION_THRESHOLD')

    with codecs.open(tweet_tokens_embs_path, 'r', encoding='ascii') as inp_file:
        num_lines = 0
        for line in inp_file:
            num_lines += 1
            vector_strings = line.strip().split('\t')
            num_vectors = len(vector_strings)
            if num_vectors > max_tweet_length:
                max_tweet_length = num_vectors
    print('maxlen', max_tweet_length)
    data_numpy_matrix = np.zeros(shape=(num_lines, max_tweet_length, embedding_size), dtype=np.float64)

    with codecs.open(tweet_tokens_embs_path, 'r', encoding='ascii') as inp_file:
        for line_id, line in enumerate(inp_file):
            tweet_embeddings = np.zeros(shape=(max_tweet_length, embedding_size), dtype=np.float64)
            vector_strings = line.strip().split('\t')
            for token_id, vector_str in enumerate(vector_strings):
                numpy_token_embedding = str_to_embedding(vector_str)
                tweet_embeddings[token_id] = numpy_token_embedding
            data_numpy_matrix[line_id] = tweet_embeddings

    y_s = pd.read_csv(labels_filepath, sep="\t", names=['class', 'text'], quoting=3)['class'].values
    print(y_s.shape)
    print(data_numpy_matrix.shape)
    X_train = data_numpy_matrix[:test_start_line_id]
    print('X train', X_train.shape)
    X_test = data_numpy_matrix[test_start_line_id:dev_start_line_id]
    print('X dev', X_test.shape)
    X_dev = data_numpy_matrix[dev_start_line_id:]
    print('X dev', X_dev.shape)
    del data_numpy_matrix
    y_train = y_s[:test_start_line_id]
    y_test = y_s[test_start_line_id:dev_start_line_id]
    y_dev = y_s[dev_start_line_id:]
    print('y train', y_train.shape)
    print('y test', y_test.shape)
    print('y dev', y_dev.shape)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     data_numpy_matrix, y_s, test_size=test_size, random_state=42)
    # del data_numpy_matrix
    # X_train, X_dev, y_train, y_dev = train_test_split(
    #     X_train, y_train, test_size=dev_size, random_state=42)

    model = TextCNNWithDynamicEmbeddings(max_tweet_length)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'], )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', )
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[early_stopping, ],
              validation_data=(X_dev, y_dev))

    predicted_test_prob = model.predict(X_test)
    predicted_test_labels = []
    predicted_dev_prob = model.predict(X_dev)
    predicted_dev_labels = []

    for subarray in predicted_test_prob:
        label = 1 if subarray[0] >= decision_threshold else 0
        predicted_test_labels.append(label)

    for subarray in predicted_dev_prob:
        label = 1 if subarray[0] >= decision_threshold else 0
        predicted_dev_labels.append(label)

    dev_precision = precision_score(y_dev, predicted_dev_labels, )
    dev_recall = recall_score(y_dev, predicted_dev_labels, )
    dev_f_measure = f1_score(y_dev, predicted_dev_labels, )
    print(f"Dev:\n\tPrecision: {dev_precision}\n"
          f"\tRecall: {dev_recall}\n\tF-measure: {dev_f_measure}")
    with open(dev_results_path, "a+", encoding="utf-8") as dev_res_file:
        dev_res_file.write(f"{round(dev_precision, 4)},{round(dev_recall, 4)},"
                           f"{round(dev_f_measure, 4)}\n")
    test_precision = precision_score(y_test, predicted_test_labels, )
    test_recall = recall_score(y_test, predicted_test_labels, )
    test_f_measure = f1_score(y_test, predicted_test_labels, )
    print(f"Test:\n\tPrecision: {test_precision}\n"
          f"\tRecall: {test_recall}\n\tF-measure: {test_f_measure}")
    with open(test_results_path, "a+", encoding="utf-8") as test_res_file:
        test_res_file.write(f"{round(test_precision, 4)},{round(test_recall, 4)},"
                            f"{round(test_f_measure, 4)}\n")
    print(classification_report(y_test, predicted_test_labels))


if __name__ == '__main__':
    main()

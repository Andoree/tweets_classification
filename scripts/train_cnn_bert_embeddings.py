import codecs
import os

import numpy as np
import pandas as pd
import yaml
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


def get_tweets_maxlen_n_lines(tweet_tokens_embs_paths_list):
    max_tweet_length = -1
    line_numbers = []
    for tweet_tokens_emb_path in tweet_tokens_embs_paths_list:
        with codecs.open(tweet_tokens_emb_path, 'r', encoding='ascii') as inp_file:
            num_lines = 0
            for line in inp_file:
                num_lines += 1
                vector_strings = line.strip().split('\t')
                num_vectors = len(vector_strings)
                if num_vectors > max_tweet_length:
                    max_tweet_length = num_vectors
            line_numbers.append(num_lines)
    return max_tweet_length, line_numbers


def init_embeddings_matrix(tweets_embeddings_matrix, tweet_tokens_embs_paths_list,
                           max_tweet_length, embedding_size, line_numbers):
    for i, tweet_tokens_emb_path in enumerate(tweet_tokens_embs_paths_list):
        offset = sum(line_numbers[:i]) if i > 0 else 0
        with codecs.open(tweet_tokens_emb_path, 'r', encoding='ascii') as inp_file:
            for line_id, line in enumerate(inp_file):
                tweet_embeddings = np.zeros(shape=(max_tweet_length, embedding_size), dtype=np.float64)
                vector_strings = line.strip().split('\t')
                for token_id, vector_str in enumerate(vector_strings):
                    numpy_token_embedding = str_to_embedding(vector_str)
                    tweet_embeddings[token_id] = numpy_token_embedding
                tweets_embeddings_matrix[offset + line_id] = tweet_embeddings
            assert line_id + 1 == line_numbers[i]


def create_X_y(tweets_embeddings_matrix, tweet_tokens_embs_paths_list, labels_filepaths_list,
               test_start_line_ids_list,
               dev_start_line_ids_list, negative_examples_ratios_list, line_numbers):
    """
    Plan:
        1) Open labels file for each language.
        2) Create global row_id column. Separate train, test, dev
        3) Get test embeddings using 2 offsets: language-level
        and train/test/dev-level
        4) Sample positive and negative labels using proportion
        5) Concatenate positive and negative examples
        6) Get row_ids.values of concatenated train and dev labels
        7) Sample train and dev embeddings using this row_ids
    """
    y_train_list = []
    y_test_list = []
    y_dev_list = []
    X_test_list = []

    for i, this_language_tweet_tokens_emb_path in enumerate(tweet_tokens_embs_paths_list):
        language_offset = sum(line_numbers[:i]) if i > 0 else 0
        this_language_test_start = test_start_line_ids_list[i]
        this_language_dev_start = dev_start_line_ids_list[i]
        this_language_negative_ratio = negative_examples_ratios_list[i]
        this_language_labels_tsv_path = labels_filepaths_list[i]
        # 1. Open labels file for each language
        y_s_df = pd.read_csv(this_language_labels_tsv_path, sep="\t",
                             names=['class', 'text'], quoting=3)
        num_examples = y_s_df.shape[0]
        # 2. Create global row_id column. Separate train, test, dev
        row_ids = [language_offset + i for i in range(num_examples)]
        y_s_df['row_id'] = row_ids
        y_test = y_s_df[this_language_test_start
                        :this_language_dev_start]
        y_dev = y_s_df[this_language_dev_start:]
        y_test_list.append(y_test)
        y_dev_list.append(y_dev)
        # 3. Get test and dev embeddings using 2 offsets: language-level
        # and train/test/dev-level
        X_test = tweets_embeddings_matrix[language_offset + this_language_test_start
                                          :language_offset + this_language_dev_start]
        X_test_list.append(X_test)
        # 4. Sample positive and negative labels using proportion
        train_y_s_df = y_s_df.iloc[:this_language_test_start]
        positive_train_y_s_df = train_y_s_df[train_y_s_df['class'] == 1]
        negative_train_y_s_df = train_y_s_df[train_y_s_df['class'] == 0]
        num_positive_examples = positive_train_y_s_df.shape[0]
        if negative_examples_ratios_list != -1:
            num_negative_examples = num_positive_examples * this_language_negative_ratio
            negative_train_y_s_df = negative_train_y_s_df.sample(num_negative_examples)
        # 5. Concatenate positive and negative examples
        y_train_df = pd.concat([positive_train_y_s_df, negative_train_y_s_df]).sample(frac=1)
        y_train_list.append(y_train_df)
    merged_train_y_s = pd.concat(y_train_list).sample(frac=1)
    merged_y_dev = pd.concat(y_dev_list)
    # 6. Get row_ids.values of concatenated train and dev labels
    merged_train_row_ids = merged_train_y_s['row_id'].values
    merged_dev_row_ids = merged_y_dev['row_id'].values
    # 7. Sample train and dev embeddings using this row_ids
    merged_X_train = tweets_embeddings_matrix[merged_train_row_ids]
    merged_X_dev = tweets_embeddings_matrix[merged_dev_row_ids]
    del tweets_embeddings_matrix
    y_test_list = [x['class'].values for x in y_test_list]
    result = {
        'X_train': merged_X_train,
        'y_train': merged_train_y_s['class'].values,
        'X_test': X_test_list,
        'y_test': y_test_list,
        'X_dev': merged_X_dev,
        'y_dev': merged_y_dev['class'].values
    }
    return result


def predict_evaluate(model, X_test, decision_threshold, y_true):
    predicted_test_probs = model.predict(X_test)
    y_pred = []
    for subarray in predicted_test_probs:
        label = 1 if subarray[0] >= decision_threshold else 0
        y_pred.append(label)
    precision = precision_score(y_true, y_pred, )
    recall = recall_score(y_true, y_pred, )
    f_measure = f1_score(y_true, y_pred, )
    return precision, recall, f_measure, y_pred


def main():
    with open('config_train_cnn_bert_emb.ini', 'r') as inp:
        config = yaml.safe_load(inp)

    tweet_tokens_embs_paths_list = config.get('TWEET_TOKENS_EMBS_PATHS')
    test_start_line_ids_list = [int(x) for x in config.get('TEST_START_LINE_IDS')]
    dev_start_line_ids_list = [int(x) for x in config.get('DEV_START_LINE_IDS')]
    negative_examples_ratios_list = [int(x) for x in config.get('NEGATIVE_EXAMPLES_RATIOS')]
    labels_filepaths_list = config.get('LABELS_FILES')
    results_dir = config.get('RESULTS_DIR')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    test_results_fname = config.get('TEST_RESULTS_FNAME')
    dev_results_fname = config.get('DEV_RESULTS_FNAME')
    test_results_path = os.path.join(results_dir, test_results_fname)
    dev_results_path = os.path.join(results_dir, dev_results_fname)
    embedding_size = int(config.get('EMBEDDING_SIZE'))
    max_tweet_length = int(config.get('MAX_TWEET_LENGTH'))
    batch_size = int(config.get('BATCH_SIZE'))
    num_epochs = int(config.get('NUM_EPOCHS'))
    decision_threshold = float(config.get('DECISION_THRESHOLD'))

    max_tweet_length_, line_numbers = get_tweets_maxlen_n_lines(tweet_tokens_embs_paths_list)
    max_tweet_length = min(max_tweet_length, max_tweet_length_) \
        if max_tweet_length != -1 else max_tweet_length_

    overall_num_lines = sum(line_numbers)
    tweets_embeddings_matrix = np.zeros(shape=(overall_num_lines, max_tweet_length, embedding_size), dtype=np.float64)

    init_embeddings_matrix(tweets_embeddings_matrix=tweets_embeddings_matrix,
                           tweet_tokens_embs_paths_list=tweet_tokens_embs_paths_list,
                           max_tweet_length=max_tweet_length, embedding_size=embedding_size,
                           line_numbers=line_numbers)

    data_dictionary = create_X_y(tweets_embeddings_matrix=tweets_embeddings_matrix,
                                 tweet_tokens_embs_paths_list=tweet_tokens_embs_paths_list,
                                 test_start_line_ids_list=test_start_line_ids_list,
                                 dev_start_line_ids_list=dev_start_line_ids_list,
                                 negative_examples_ratios_list=negative_examples_ratios_list,
                                 line_numbers=line_numbers, labels_filepaths_list=labels_filepaths_list)
    X_train = data_dictionary['X_train']
    y_train = data_dictionary['y_train']
    X_dev = data_dictionary['X_dev']
    y_dev = data_dictionary['y_dev']
    X_test_list = data_dictionary['X_test']
    true_y_test_list = data_dictionary['y_test']
    print('X train', X_train.shape)
    print('y train', y_train.shape)
    print('X dev', X_dev.shape)
    print('y dev', y_dev.shape)

    model = TextCNNWithDynamicEmbeddings(max_tweet_length)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'], )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', )
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[early_stopping, ],
              validation_data=(X_dev, y_dev))

    dev_precision, dev_recall, dev_f_measure, _ = \
        predict_evaluate(model, X_test=X_dev, decision_threshold=decision_threshold, y_true=y_dev)
    print(f"Dev:\n\tPrecision: {dev_precision}\n"
          f"\tRecall: {dev_recall}\n\tF-measure: {dev_f_measure}")
    with open(dev_results_path, "a+", encoding="utf-8") as dev_res_file:
        dev_res_file.write(f"{round(dev_precision, 4)},{round(dev_recall, 4)},"
                           f"{round(dev_f_measure, 4)}\n")

    for i in range(len(tweet_tokens_embs_paths_list)):
        one_lang_X_test = X_test_list[i]
        one_lang_y_test_true = true_y_test_list[i]
        print('X test', one_lang_X_test.shape)
        print('y test', one_lang_y_test_true.shape)


        test_precision, test_recall, test_f_measure, test_y_pred = \
            predict_evaluate(model, X_test=one_lang_X_test, decision_threshold=decision_threshold,
                             y_true=one_lang_y_test_true)
        print(f"Test:\n\tPrecision: {test_precision}\n"
              f"\tRecall: {test_recall}\n\tF-measure: {test_f_measure}")
        with open(test_results_path, "a+", encoding="utf-8") as test_res_file:
            test_res_file.write(f"{round(test_precision, 4)},{round(test_recall, 4)},"
                                f"{round(test_f_measure, 4)}\n")
        print(classification_report(one_lang_y_test_true, test_y_pred))


if __name__ == '__main__':
    main()

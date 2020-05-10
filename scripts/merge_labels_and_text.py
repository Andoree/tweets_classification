import os

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    test_labels_path = r"../corpora/not_split_all_eng_tweets/test/labels.txt"
    test_texts_path = r"../corpora/not_split_all_eng_tweets/test/text.txt"
    train_labels_path = r"../corpora/not_split_all_eng_tweets/train/labels.txt"
    train_texts_path = r"../corpora/not_split_all_eng_tweets/train/text.txt"
    result_dir = "../new_corpora/corpus_en_ones/"
    negative_proportion = 0
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    test_labels_df = pd.read_csv(test_labels_path, encoding="utf-8", names=['class'])
    test_texts = []
    with open(test_texts_path, "r", encoding="utf-8") as inp:
        for line in inp:
            test_texts.append(line.strip())
    test_texts = numpy.array(test_texts)
    test_labels_df['tweet'] = test_texts

    train_labels_df = pd.read_csv(train_labels_path, encoding="utf-8", names=['class'])
    train_texts = []
    with open(train_texts_path, "r", encoding="utf-8") as inp:
        for line in inp:
            train_texts.append(line.strip())
    train_texts = numpy.array(train_texts)
    train_labels_df['tweet'] = train_texts
    train_labels_df = pd.concat([train_labels_df, test_labels_df])
    train_df, test_df, _, _ = \
        train_test_split(train_labels_df, train_labels_df, test_size=0.2, random_state=42)
    train_df, dev_df, _, _ = \
        train_test_split(train_df, train_df, test_size=0.1, random_state=42)

    train_positive_class_df = train_df[train_df['class'] == 1]
    train_negative_class_df = train_df[train_df['class'] == 0]
    num_positive_examples = train_positive_class_df.shape[0]
    num_negative_examples = num_positive_examples * negative_proportion
    train_negative_class_df = train_negative_class_df.sample(num_negative_examples, )
    normalized_train_df = pd.concat([train_positive_class_df, train_negative_class_df]).sample(frac=1)


    # train_positive_class_df = train_labels_df[train_labels_df['class'] == 1]
    # train_negative_class_df = train_labels_df[train_labels_df['class'] == 0]
    # num_positive_examples = train_positive_class_df.shape[0]
    # num_negative_examples = num_positive_examples * negative_proportion
    # train_negative_class_df = train_negative_class_df.sample(num_negative_examples, )
    # class_normalized_train_df = pd.concat([train_positive_class_df, train_negative_class_df]).sample(frac=1)

    output_train_path = os.path.join(result_dir, "train.tsv")
    output_test_path = os.path.join(result_dir, "test.tsv")
    output_dev_path = os.path.join(result_dir, "dev.tsv")
    normalized_train_df.to_csv(output_train_path, sep='\t', encoding="utf-8", index=False, header=None)
    test_df.to_csv(output_test_path, sep='\t', encoding="utf-8", index=False)
    dev_df.to_csv(output_dev_path, sep='\t', encoding="utf-8", index=False, header=None)
    print(normalized_train_df)


if __name__ == '__main__':
    main()

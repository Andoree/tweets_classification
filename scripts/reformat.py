import os

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_ORDER = ["class", "tweet"]


def main():
    training_path = r"../corpora/not_split_all_ru_tweets/task2_ru_training.tsv"
    val_path = r"../corpora/not_split_all_ru_tweets/task2_ru_validation.tsv"
    out_dir = r"../competition_data/adr_reviews/"
    negative_proportion = 5
    output_format = 'csv'
    # training_path = r"corpus_full_eng/all_tweets_ruen.tsv"
    # val_path = r"corpus_full_eng/test.tsv"
    # out_dir = r"english_tweets_normalized"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_df = pd.read_csv(training_path, sep="\t", encoding="utf-8")
    train_df = train_df[COLUMN_ORDER]
    train_df.drop_duplicates(inplace=True)

    test_df = pd.read_csv(val_path, sep="\t", encoding="utf-8")
    test_df = test_df[COLUMN_ORDER]
    test_df.drop_duplicates(inplace=True)
    train_df, dev_df, _, _ = \
        train_test_split(train_df, train_df, test_size=0.1, random_state=42)
    if negative_proportion != -1:
        train_positive_class_df = train_df[train_df['class'] == 1]
        train_negative_class_df = train_df[train_df['class'] == 0]
        num_positive_examples = train_positive_class_df.shape[0]
        num_negative_examples = num_positive_examples * negative_proportion
        train_negative_class_df = train_negative_class_df.sample(num_negative_examples, )
        class_normalized_train_df = pd.concat([train_positive_class_df, train_negative_class_df]).sample(frac=1)
    else:
        class_normalized_train_df = train_df.sample(frac=1)

    out_train_path = os.path.join(out_dir, f"train.{output_format}")
    out_test_path = os.path.join(out_dir, f"test.{output_format}")
    out_dev_path = os.path.join(out_dir, f"dev.{output_format}")

    if output_format == 'tsv':
        class_normalized_train_df.to_csv(out_train_path, sep="\t", encoding="utf-8", quoting=3, index=False,
                                         quotechar=None)
        test_df.to_csv(out_test_path, sep="\t", encoding="utf-8", index=False, quoting=3, quotechar=None)
        dev_df.to_csv(out_dev_path, sep="\t", encoding="utf-8", index=False, quoting=3, quotechar=None)
    elif output_format == 'csv':
        test_df = test_df[['tweet', 'class']]
        class_normalized_train_df.to_csv(out_train_path, encoding="utf-8", index=False, )
        test_df.to_csv(out_test_path, encoding="utf-8", index=False, )
        dev_df.to_csv(out_dev_path, encoding="utf-8", index=False, )


if __name__ == '__main__':
    main()

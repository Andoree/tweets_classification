import os
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_ORDER = ["class", "tweet"]


def main():
    training_path = r"corpora/not_split_all_ru_tweets/task2_ru_training.tsv"
    val_path = r"corpora/not_split_all_ru_tweets/task2_ru_validation.tsv"
    out_dir = r"corpora/corpus_full_ru___/"
    # training_path = r"corpus_full_eng/train.tsv"
    # val_path = r"corpus_full_eng/test.tsv"
    # out_dir = r"english_tweets_normalized"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_df = pd.read_csv(training_path, sep="\t", encoding="utf-8")
    train_df = train_df[COLUMN_ORDER]

    test_df = pd.read_csv(val_path, sep="\t", encoding="utf-8")
    test_df = test_df[COLUMN_ORDER]
    train_df, dev_df, _, _ = \
        train_test_split(train_df, train_df, test_size=0.1, random_state=42)

    out_train_path = os.path.join(out_dir, "train.tsv")
    out_test_path = os.path.join(out_dir, "test.tsv")
    out_dev_path = os.path.join(out_dir, "dev.tsv")

    train_df.to_csv(out_train_path, sep="\t", encoding="utf-8", quoting=3, index=False, quotechar=None)
    test_df.to_csv(out_test_path, sep="\t", encoding="utf-8", index=False, quoting=3, quotechar=None)
    dev_df.to_csv(out_dev_path, sep="\t", encoding="utf-8", index=False, quoting=3,quotechar=None)

    print(train_df)
if __name__ == '__main__':
    main()

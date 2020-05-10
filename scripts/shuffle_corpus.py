import os
import pandas as pd


def main():
    result_dir = r"../new_corpora/corpus_ruen_normalized/"
    train_path = os.path.join(result_dir, "train.tsv")
    test_path = os.path.join(result_dir, "test.tsv")
    dev_path = os.path.join(result_dir, "dev.tsv")
    train_df = pd.read_csv(train_path, sep="\t", encoding="utf-8", quoting=3,header=None)
    print(train_df)
    train_df = train_df.sample(frac=1)
    train_df.to_csv(train_path,sep='\t', encoding="utf-8", quoting=3,index=False)

    test_df = pd.read_csv(test_path, sep="\t", encoding="utf-8", quoting=3)
    test_df = test_df.sample(frac=1)
    test_df.to_csv(test_path, sep='\t', encoding="utf-8",quoting=3, index=False)
    print(test_df)
    dev_df = pd.read_csv(dev_path, sep="\t", encoding="utf-8", quoting=3,header=None)
    dev_df = dev_df.sample(frac=1)
    dev_df.to_csv(dev_path, sep='\t', encoding="utf-8",quoting=3, index=False)
    print(dev_df)

if __name__ == '__main__':
    main()

import pandas as pd


def main():
    input_path = r"../new_corpora/all_tweets_ruen/all_tweets_ruen.tsv"
    output_path = r"../new_corpora/all_tweets_ruen/all_tweets_no_labels.txt"
    tweets_df = pd.read_csv(input_path, sep="\t", names=['id', 'text'], quoting=3)
    print(tweets_df)
    tweets_df.drop(columns=['id'], axis=1, inplace=True)
    tweets_df.to_csv(output_path, header=False, index=False, quoting=3, sep="\t")


if __name__ == '__main__':
    main()

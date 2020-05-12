import pandas as pd

def main():
    old_tweets_path = r"../hlp_workshop_test_set_goldstandard.txt"
    output_path = r"../new_corpora/new_old_test.tsv"
    tweets_df = pd.read_csv(old_tweets_path, sep="\t", names=['id', 'tweet', 'class'])
    tweets_df = tweets_df[['class', 'tweet']]
    tweets_df.to_csv(output_path, sep='\t', index=False)
    print(tweets_df)

    pass


if __name__ == '__main__':
    main()

import pandas as pd


def main():
    covid_tweets_path = r"../covid_tweets.csv"
    adr_probs_path = r"../covid_adr_probs.tsv"
    output_path = "../covid_tweets_with_adr_prob.csv"
    covid_tweets_df = pd.read_csv(covid_tweets_path, )
    print('covid_tweets_df', covid_tweets_df)
    adr_probs_df = pd.read_csv(adr_probs_path, sep="\t", names=["negative_class", "adr_prob"])
    print('adr_probs_df', adr_probs_df)
    adr_positive_class_probs = adr_probs_df.adr_prob
    print('adr_probs_positive_df', adr_positive_class_probs)
    covid_tweets_df['adr_prob'] = adr_positive_class_probs
    print('covid_tweets_df', covid_tweets_df)
    covid_tweets_df.sort_values(by='adr_prob', inplace=True, ascending=False)
    print('covid_tweets_df', covid_tweets_df)
    covid_tweets_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()

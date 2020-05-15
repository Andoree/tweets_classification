import pandas as pd


def main():
    tweets_scores_path = r"../covid_prob_scores.csv"
    output_path = r"../adr_covid_prob_scores.csv"
    tweet_scores_df = pd.read_csv(tweets_scores_path)
    tweet_scores_df.drop(['p_DI', 'Finding', 'p_EF', 'p_INF'], inplace=True)
    tweet_scores_df.to_csv(output_path, compression='zip')


if __name__ == '__main__':
    main()
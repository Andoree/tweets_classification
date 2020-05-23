import pandas as pd


def main():
    covid_tweets_path = r"../covid_tweets.csv"
    output_path = "../cleaned_covid_tweets.csv"
    covid_tweets_df = pd.read_csv(covid_tweets_path, )

    covid_tweets_df.text = covid_tweets_df.text.apply(lambda x: x.replace("\n", " ").replace("\t", " "))
    covid_tweets_df.drop(['date', 'resource'], axis=1, inplace=True)
    covid_tweets_df = covid_tweets_df[['text','id']]
    covid_tweets_df.to_csv(output_path, index=False,)



if __name__ == '__main__':
    main()

import json
import unicodedata
import nltk
# Isn't this second import double after import nltk?:
from nltk.tokenize.toktok import ToktokTokenizer
import re
import os
import pickle

# For SentiStrength
import subprocess
import shlex
from sentistrength import PySentiStr

# For Pattern
from pattern.nl import sentiment

import numpy as np
import pandas as pd

import spacy
# from spacy.lemmatizer import Lemmatizer
from spacy.lang.nl.stop_words import STOP_WORDS

from tqdm import tqdm_notebook as tqdm
from pprint import pprint


def calc_sentiments(input):
    result_dual = senti.getSentiment(input, score='dual')
    return result_dual

    # result_binary = senti.getSentiment(input, score='binary')
    # print(result_binary)
    #
    # result_trinary = senti.getSentiment(input, score='trinary')
    # print(result_trinary)


def RateSentiment(sentiString, SentiStrengthLocationIn, SentiStrengthLanguageFolderIn):
    # open a subprocess using shlex to get the command line string into the correct args list format
    p = subprocess.Popen(
        shlex.split(
            "java -jar '" + SentiStrengthLocationIn + "' stdin sentidata '" + SentiStrengthLanguageFolderIn + "'"),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # communicate via stdin the string to be rated. Note that all spaces are replaced with +
    b = bytes(sentiString.replace(" ", "+"), 'utf-8')  # Can't send string in Python 3, must send bytes
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")  # convert from byte
    stdout_text = stdout_text.rstrip().replace("\t",
                                               " ")  # remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    return stdout_text + " " + sentiString


if __name__ == "__main__":
    # freeze_support() ? -> https://github.com/RaRe-Technologies/gensim/issues/940

    # Global parameters
    rem_line_break = True
    replace_accented_char = True
    lower_case = True
    rem_spec_char = True
    rem_digits = True
    rem_stopwords = True
    rem_urls = True
    rem_short_words = True  # This is only performed if rem_stopwords = True
    lemmatize = True

    # setup SentiStrength
    senti = PySentiStr()

    # Set path to SentiStrength files
    SentiStrengthLocation = "C:/Users/mila1/Downloads/SentiStrengthCom.jar"  # The location of SentiStrength on your computer
    SentiStrengthLanguageFolder = "C:/SentiStrength_Data_Dutch/"  # The location of the unzipped SentiStrength data files on your computer

    senti.setSentiStrengthPath("C:/Users/mila1/Downloads/SentiStrengthCom.jar")
    senti.setSentiStrengthLanguageFolderPath("C:/SentiStrength_Data_Dutch/")

    # Data Folder
    # Add "_test" to the end of this dir name to use a smaller test dataset
    year = "2021"
    data_dir = "../nitrogentweets_{}/".format(year)
    results_dir = "../results/"
    version_number = "17_{}".format(year[-2:])



    # Set up Spacy
    nlp = spacy.load('nl_core_news_sm')

    # Create arrays with all tweets and their metadata
    tweet_author_array = []
    tweet_conversation_id_array = []
    tweet_created_at_array = []
    tweets_array = []
    tweet_text_original_array = []
    date_array = []
    hashtags_array = []
    # public_metrics_array = []
    like_count_array = []
    quote_count_array = []
    reply_count_array = []
    retweet_count_array = []
    mentions_array = []
    urls_array = []
    pattern_senti_array = []
    pattern_subjectivity_array = []

    # List of stopwords
    stopword_list = nltk.corpus.stopwords.words('dutch')
    additional_stopwords = ['stikstof', 'probleem', 'goed', 'stikstofcrisis', 'stikstofprobleem', 'willen', 'wil',
                            'wilt', 'zal', 'zullen', 'maak', 'maken', 'heel', 'echt', 'gaan', 'per']

    stopword_list.extend(additional_stopwords)

    # Total amount of files
    tot_files = len([name for name in os.listdir(data_dir)])

    j = 0
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            if j % 50 == 0:
                print("Reading file {} of {}...".format(j, tot_files))

            with open(data_dir + filename) as f:
                data = json.load(f)
            j += 1

            # If you want to print a single tweet:
            # print(data['data'][5]['text'])

            tweets = data['data']
            # full_tweet_text = ''

            for i in range(len(tweets)):
                # Clear lists for hashtags, mentions and urls
                tweet_hashtags = []
                tweet_mentions = []
                tweet_urls = []

                # Retrieve information per tweet
                ########## THIS IS WHERE OTHER INFORMATION FROM THE TWEET CAN BE COLLECTED ##########
                tweet_author = tweets[i]["author_id"]
                tweet_conversation_id = tweets[i]["conversation_id"]
                tweet_created_at = tweets[i]["created_at"]
                tweet_text = tweets[i]['text']
                tweet_text_original = tweets[i]['text']
                tweet_date = tweets[i]['created_at']
                tweet_public_metrics = tweets[i]['public_metrics']
                tweet_like_count = tweets[i]['public_metrics']['like_count']
                tweet_quote_count = tweets[i]['public_metrics']['quote_count']
                tweet_reply_count = tweets[i]['public_metrics']['reply_count']
                tweet_retweet_count = tweets[i]['public_metrics']['retweet_count']

                if 'entities' in tweets[i]:
                    if 'hashtags' in tweets[i]['entities']:
                        tweet_hashtags = tweets[i]['entities']['hashtags']

                    if 'mentions' in tweets[i]['entities']:
                        tweet_mentions = tweets[i]['entities']['mentions']

                    if 'urls' in tweets[i]['entities']:
                        tweet_urls = tweets[i]['entities']['urls']

                if rem_line_break:
                    # Remove /n
                    tweet_text = tweet_text.replace('\n', ' ')

                if replace_accented_char:
                    tweet_text = unicodedata.normalize('NFKD', tweet_text).encode('ascii', 'ignore').decode('utf-8',
                                                                                                            'ignore')

                if lower_case:
                    tweet_text = tweet_text.lower()

                # maybe remove urls before removing special characters?
                # TODO: Do not remove hastags (#)
                if rem_spec_char:
                    pattern = r'[^a-zA-z0-9\s]' if not rem_digits else r'[^a-zA-z\s]'
                    tweet_text = re.sub(pattern, '', tweet_text)

                # print("Before lemma:")
                # print(tweet_text)

                # Lemmatize
                if lemmatize:
                    doc = nlp(tweet_text)

                    # tweet_text = [token.lemma_ for token in doc]
                    tweet_text = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1])

                # print("After lemma:")
                # print(tweet_text)

                if rem_stopwords:
                    tokenizer = ToktokTokenizer()

                    tokens = tokenizer.tokenize(tweet_text)

                    tokens = [token.strip() for token in tokens]

                    filtered_tokens = [token for token in tokens if token not in stopword_list]

                    if rem_short_words:
                        filtered_tokens = [token for token in filtered_tokens if len(token) > 2]

                    tweet_text = ' '.join(filtered_tokens)
                    # senti_text = '+'.join(filtered_tokens)

                # full_tweet_text = full_tweet_text + ' ' + tweet_text

                tweet_author_array.append(tweet_author)
                tweet_conversation_id_array.append(tweet_conversation_id)
                tweet_created_at_array.append(tweet_created_at)
                tweets_array.append(tweet_text)
                tweet_text_original_array.append(tweet_text_original)
                date_array.append(tweet_date)
                hashtags_array.append(tweet_hashtags)
                # public_metrics_array.append(tweet_public_metrics)
                like_count_array.append(tweet_like_count)
                quote_count_array.append(tweet_quote_count)
                reply_count_array.append(tweet_reply_count)
                retweet_count_array.append(tweet_retweet_count)
                mentions_array.append(tweet_mentions)
                urls_array.append(tweet_urls)
                pattern_senti_array.append(sentiment(tweet_text_original)[0])
                pattern_subjectivity_array.append(sentiment(tweet_text_original)[1])
        else:
            continue

    # Calculate Sentiment with SentiStrength
    # The called method returns a list of Tuples
    # sentiments = calc_sentiments(tweets_array)
    sentiments = calc_sentiments(tweet_text_original_array)

    senti_df = pd.DataFrame(sentiments, columns=["pos_senti", "neg_senti"])
    print("lengths:")
    print("senti_df: {}".format(len(senti_df)))
    print("tweet_author_array: {}".format(len(tweet_author_array)))
    print("tweet_conversation_id_array: {}".format(len(tweet_conversation_id_array)))
    print("tweet_created_at_array: {}".format(len(tweet_created_at_array)))
    print("tweets_array: {}".format(len(tweets_array)))
    print("tweet_text_original_array: {}".format(len(tweet_text_original_array)))
    print("date_array: {}".format(len(date_array)))
    print("hashtags_array: {}".format(len(hashtags_array)))
    # print("public_metrics_array: {}".format(len(public_metrics_array)))
    print("like_count_array: {}".format(len(like_count_array)))
    print("quote_count_array: {}".format(len(quote_count_array)))
    print("reply_count_array: {}".format(len(reply_count_array)))
    print("retweet_count_array: {}".format(len(retweet_count_array)))
    print("mentions_array: {}".format(len(mentions_array)))
    print("urls_array: {}".format(len(urls_array)))
    print("pattern_senti_array: {}".format(len(pattern_senti_array)))
    print("pattern_subjectivity_array: {}".format(len(pattern_subjectivity_array)))

    # create dataframe from lists
    metadata_dict = {'date': date_array, 'tweet_author': tweet_author_array,
                     'tweet_conversation_id': tweet_conversation_id_array,
                     'tweet_created_at': tweet_created_at_array, 'tweet': tweets_array,
                     'tweet_text_original': tweet_text_original_array,
                     'hashtags': hashtags_array, 'like_count': like_count_array, 'quote_count': quote_count_array,
                     'reply_count': reply_count_array,
                     'retweet_count': retweet_count_array, 'urls': urls_array, 'pattern_senti': pattern_senti_array,
                     'pattern_subjectivity': pattern_subjectivity_array}

    metadata_df = pd.DataFrame(metadata_dict)

    # print(full_tweet_text)
    # data_dict = pd.json_normalize(data['text'])

    # # Convert the string to list

    df_tweets = pd.DataFrame(tweets_array, columns=['processed_tweets'])
    # print(df_tweets.head())

    # Tokenize tweets
    tokenizer = ToktokTokenizer()
    tokens = []

    tokenize_lambda = lambda x: tokenizer.tokenize(x)

    df_tokenized_tweets = df_tweets['processed_tweets'].apply(tokenize_lambda).to_frame()
    df_tokenized_tweets.columns = ['tokenized_tweets']
    print("df_tokenized_tweets: {}".format(len(df_tokenized_tweets)))

    df_tweets_senti = metadata_df.join(senti_df).join(df_tokenized_tweets)
    print("df_tweets_senti: {}".format(len(df_tweets_senti)))

    ###### WRITE DATASET TO JSON FILE ######
    # Remove 'T' and timezone from timestamp string
    df_tweets_senti['date'] = df_tweets_senti['date'].str.replace(r'T', ' ')
    df_tweets_senti['date'] = df_tweets_senti['date'].str[:-5]

    # converting the string to datetime format
    df_tweets_senti['date'] = pd.to_datetime(df_tweets_senti['date'], format='%Y-%m-%d %H:%M:%S')

    # Remove rows that were split up in the processing
    print("Length before removing split up rows: {}".format(len(df_tweets_senti)))
    df_tweets_senti[['pos_senti']] = df_tweets_senti[['pos_senti']].apply(pd.to_numeric, errors='coerce')

    # Keep only the rows with the right data typs
    df_tweets_senti = df_tweets_senti[df_tweets_senti['pos_senti'].notna()]
    print("Length after removing split up rows: {}".format(len(df_tweets_senti)))

    df_tweets_senti.to_csv(results_dir + 'set_results_V' + version_number + '.csv', index=False)
    # df_tweets_senti.to_csv(results_dir + 'set_results_V' + version_number + '.csv', index=False)

    with open(results_dir + 'set_results_V' + version_number, 'wb') as f:
        pickle.dump(df_tweets_senti, f)

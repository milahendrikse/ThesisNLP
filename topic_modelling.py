import pandas as pd
import pickle
import re
import statistics
import gensim
import numpy as np

# Natural Language Processing (NLP)
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


def topic_modelling_prep(df_tweets_senti_in, version_number_in, time_window_in, sentiment_in, slice_id):
    # Create a id2word dictionary
    id2word = Dictionary(df_tweets_senti_in['tokenized_tweets'])

    # Filtering Extremes
    # no_below kan omhoog! We hebben veel tweets
    # No_above: probeer omlaag te halen
    id2word.filter_extremes(no_below=2, no_above=.50)
    # print(len(id2word))
    # with open(
    #         dictionaries_dir + 'dictionary_V{}_tw{}_{}_slice{}'.format(version_number_in, time_window_in, sentiment_in,
    #                                                                    slice_id), 'wb') as f:
    #     pickle. (id2word, f)

    # Creating a corpus object
    corpus = [id2word.doc2bow(d) for d in df_tweets_senti_in['tokenized_tweets']]
    # with open(corpusses_dir + 'corpus_V{}_tw{}_{}_slice{}'.format(version_number_in, time_window_in, sentiment_in,
    #                                                               slice_id), 'wb') as f:
    #     pickle.dump(corpus, f)

    return id2word, corpus


def topic_modelling(id2word_in, corpus_in, num_topics_in, print_topic_in):
    # Instantiating a Base LDA model
    base_model = LdaMulticore(corpus=corpus_in, num_topics=num_topics_in, id2word=id2word_in, workers=12, passes=5)

    # # Use LDA mallet for better score..?
    # base_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_in, num_topics=num_topics_in, id2word=id2word_in)
    # with open(topic_models_dir + '/topic_model_V' + version_numberIn + "_" + sentiment, 'wb') as f:
    #     pickle.dump(base_model, f)

    if (print_topic_in):

        # Filtering for words
        words = [re.findall(r'"([^"]*)"', t[1]) for t in base_model.print_topics()]

        # Create Topics
        topics = [' '.join(t[0:10]) for t in words]

        # Getting the topics
        for id, t in enumerate(topics):
            print(f"------ Topic {id} ------")
            print(t, end="\n\n")

    return base_model


def grid_search(df_tweets_senti_in, write_version_number_in, sentiment_in, time_windows_in, num_topics_in,
                print_topic_in):

    # Create new datasets if sentiment is postive or negative
    if sentiment_in == "pos":
        df_tweets_senti_in = df_tweets_senti_in.loc[((df_tweets_senti_in['pos_senti'] > 1) & (df_tweets_senti_in['pos_senti'] > -1 * df_tweets_senti_in['neg_senti']))]
    if sentiment_in == "neg":
        df_tweets_senti_in = df_tweets_senti_in.loc[((df_tweets_senti_in['neg_senti'] < -1) & (df_tweets_senti_in['neg_senti'] < -1 * df_tweets_senti_in['pos_senti']))]

    # Create a id2word dictionary and corpus for each data slice for each time window and num_topics
    topic_modelling_prep_objects_dict = {}

    for time_window in time_windows_in:
        print("prep for time_window: {}".format(time_window))

        time_slices = df_tweets_senti_in.groupby(pd.Grouper(key="date", freq="{}D".format(time_window)))

        tm_attributes = []

        iteration = 0
        for i, data_slice in time_slices:
            # doe not save the data slice if it contains no tweets
            if len(data_slice) < 5:
                continue

            iteration += 1
            print("Prep for data slice {} out of {}".format(iteration, len(time_slices)))

            # Create Dictionary and corpus
            id2word, corpus = topic_modelling_prep(data_slice, write_version_number_in, time_window, sentiment_in,
                                                   iteration)

            tm_attributes.append([data_slice, id2word, corpus])

        topic_modelling_prep_objects_dict[time_window] = tm_attributes

    grid = [['time_window', 'num_topics', 'average_coherence']]
    grid_std = [['time_window', 'num_topics', 'std_coherence']]
    for num in num_topics_in:
        print("num_topics: {}".format(num))
        coherence_scores = []

        for time_window in time_windows_in:
            print("time_window: {}".format(time_window))

            iteration = 0
            for data_slice, id2word, corpus in topic_modelling_prep_objects_dict[time_window]:
                iteration += 1
                print("Calc base model for slice {} out of {}".format(iteration, len(topic_modelling_prep_objects_dict[time_window])))

                # create topic model with time_window and num_topics for each data slice
                base_model = topic_modelling(id2word, corpus, num, print_topic_in)

                # SAVE THE TOPIC MODEL
                with open(topic_models_dir + 'topic_model_V{}_{}_tw{}_nt{}_slice{}'.format(write_version_number_in, sentiment_in, time_window, num, iteration),
                          'wb') as f:
                    pickle.dump(base_model, f)

                # Compute Coherence Score
                coherence_model = CoherenceModel(model=base_model, texts=data_slice['tokenized_tweets'].tolist(),
                                                 dictionary=id2word, coherence='c_v')

                coherence_lda_model_base = coherence_model.get_coherence()
                coherence_scores.append(coherence_lda_model_base)

            # SAVE THE COHERENCE_SCORES LIST!
            with open(coherence_dir + 'coherence_V{}_{}_tw{}_nt{}'.format(write_version_number_in, sentiment_in, time_window, num),
                      'wb') as f:
                pickle.dump(coherence_scores, f)



            # calc average coherence of all topic models over the year for current num_topics
            av_coherence = sum(coherence_scores) / len(coherence_scores)
            std_coherence = statistics.stdev(coherence_scores)
            grid.append([time_window, num, av_coherence])
            grid_std.append([time_window, num, std_coherence])

    with open(grid_dir + 'grid_V' + write_version_number + "_" + sentiment_in, 'wb') as f:
        pickle.dump(grid, f)

    with open(grid_dir + 'grid_std_V' + write_version_number + "_" + sentiment_in, 'wb') as f:
        pickle.dump(grid_std, f)


if __name__ == "__main__":
    # Global variables
    # version 13 are all 2019 tweets
    # version 14 are all 2020 tweets
    # version 15 are all 2019 tweets
    # version 16 are all 2020 tweets
    # version 16 are all 2019 tweets ldamallet
    # version 18 are 2019, 2020 and 2021 tweets ldamallet
    # version 20 are 2019, 2020 and 2021 tweets
    # version 21 are 2019 (from June onwards), 2020 and 2021 tweets (I think this one failed)
    # version 22 are 2019 (from June onwards), 2020 and 2021 tweets, num_topics = 14, time_window = 7
    # version 22 are 2019 (from June onwards), 2020 and 2021 tweets
    # version 22 saves topic models too
    # version 27 with sentiment analysis pre-processing fix
    write_version_number = "27"
    # version 17 has sentiment analysis pre-processing fix
    read_version_number = "17"
    results_dir = "results/"
    dictionaries_dir = "dictionaries/"
    corpusses_dir = "corpusses/"
    topic_models_dir = "topic_models/"
    grid_dir = "grid/"
    coherence_dir = "coherence/"

    mallet_path = "C:/Users/mila1/Downloads/mallet-2.0.8/mallet-2.0.8"

    df_tweets_senti_19 = pickle.load(open(results_dir + 'set_results_V' + read_version_number + "_19", 'rb'))
    df_tweets_senti_20 = pickle.load(open(results_dir + 'set_results_V' + read_version_number + "_20", 'rb'))
    df_tweets_senti_21 = pickle.load(open(results_dir + 'set_results_V' + read_version_number + "_21", 'rb'))

    # REMOVE TWEETS UP UNTIL MAY, BECAUSE THERE ARE LESS THAN 1000 TWEETS PER MONTH IN THOSE MONTHS
    # This removes 1636 tweets
    start_date = "2019-06-01 00:00:01"

    after_start_date = df_tweets_senti_19["date"] >= start_date
    df_tweets_senti_19 = df_tweets_senti_19.loc[after_start_date]

    frames = [df_tweets_senti_19, df_tweets_senti_20, df_tweets_senti_21]
    df_tweets_senti = pd.concat(frames)

    print(len(df_tweets_senti))
    # df_tweets_senti.to_csv(results_dir + 'testing_nan.csv', index=False)
    # print("done")

    # ###### SPLIT DATASET ######
    # # selecting rows based on condition
    # print(new_df_tweets_senti.dtypes)
    # # print(df_tweets_senti['pos_senti'])
    # pos_dataset = new_df_tweets_senti[new_df_tweets_senti['pos_senti'] > 1]
    # neg_dataset = new_df_tweets_senti[new_df_tweets_senti['neg_senti'] < -1]
    #
    # print("len pos_senti:")
    # print(len(pos_dataset))
    #
    # print("len neg_senti:")
    # print(len(neg_dataset))
    #
    # # print(pos_dataset['pos_senti'])
    # # print(neg_dataset['neg_senti'])
    #
    # print("average pos score:")
    # print(pos_dataset["pos_senti"].mean())
    #
    # print("average neg score:")
    # print(neg_dataset["neg_senti"].mean())

    # ###### TOPIC MODELLING ######
    # # topic modelling on positive tweets
    # print("Topics in positive tweets")
    # topics_pos = topic_modelling(pos_dataset, version_number, "pos")
    #
    # print("Topics in negative tweets")
    # # topic modelling on negative tweets
    # topics_neg = topic_modelling(neg_dataset, version_number, "neg")
    #
    # print("Topics in all tweets")
    # # topic modelling on negative tweets
    # topics_all = topic_modelling(new_df_tweets_senti, version_number, "all")

    ###### GRID SEARCH ######
    # sentiment = "all"
    # Create list of datasets per time window
    time_windows = range(7, 31, 7)
    num_topics = range(2, 17, 2)
    # time_windows = range(7, 8, 7)
    # num_topics = range(14, 15, 2)
    print_topic = False

    # DICTIONARIES AND CORPUSSES ARE NOT STORED NOW
    # Run grid search for each sentiment
    grid_search(df_tweets_senti, write_version_number, "all", time_windows, num_topics, print_topic)
    # grid_search(df_tweets_senti, write_version_number, "pos", time_windows, num_topics, print_topic)
    # grid_search(df_tweets_senti, write_version_number, "neg", time_windows, num_topics, print_topic)

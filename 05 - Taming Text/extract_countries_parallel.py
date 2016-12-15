import pandas as pd
import numpy as np
import nltk
import pickle

import re
import pycountry
import jellyfish

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.sentiment import vader

from joblib import Parallel, delayed
import multiprocessing

py_countries = pycountry.countries

def regexify(text):
    return text.replace(' ', '\s+')

country_names = [
    regexify(c.name.lower())
    for c in py_countries
]

official_names = [
    regexify(c.official_name.lower())
    for c in py_countries
    if 'official_name' in c._fields
]

codes = (
    [c.alpha_2.lower() for c in py_countries] +
    [c.alpha_3.lower() for c in py_countries]
)

codes = []

all_needles = official_names + country_names + codes

reg = r'(\b(?:' + r')\b|\b(?:'.join(all_needles) + r')\b)'

def get_non_empty(l):
    return list(set([i for x in l for i in list(x) if i != '']))

def extract_countries(text, axis=None):
    m = re.findall(reg, text)
    results = get_non_empty(m)
    return [py_countries.lookup(res).name for res in results]

def analyze_sentiment_vader(text):
    analyzer = vader.SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)

def analyze_sentiment_liuhu(sentence):
    from nltk.corpus import opinion_lexicon
    from nltk.tokenize import treebank

    tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(tokenized_sent))) # x axis for the plot
    y = []

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
        elif word in opinion_lexicon.negative():
            neg_words += 1

    v = pos_words - neg_words
    v_scaled = v/len(tokenized_sent)

    return {
        'pos': pos_words,
        'neg': neg_words,
        'compound': v_scaled
    }

def get_entry_text(entry):
    subject_body = str(entry['ExtractedSubject']) + '\n\n' + str(entry['ExtractedBodyText'])
    return subject_body.replace('\n', ' ').lower()

def process_mail(mail):
    (idx, entry) = mail
    text         = get_entry_text(entry)
    countries    = extract_countries(text)
    score        = analyze_sentiment_vader(text)

    return {
        'index'     : idx,
        'sentiment' : score['compound'],
        'countries' : countries
    }

def process_mails(df):
    print('Processing mails in parallel...')
    par = Parallel(n_jobs=4, verbose=5)
    processed = par(delayed(process_mail)((idx, row)) for idx, row in df.iterrows())
    print('=> Done.')

    res = {}

    print('Aggregating the results by country...')

    for entry in processed:
        (sentiment, countries) = (entry['sentiment'], entry['countries'])
        for country in countries:
            if country not in res:
                res[country] = {
                    'sentiments': [],
                    'count': 0
                }

            res[country]['sentiments'].append(sentiment)
            res[country]['count'] += 1

    print('=> Done.')

    return res


if __name__ == '__main__':
    pickle_file = 'countries_sentiments_vader_nocodes.p'

    print('Loading data...')
    filename='hillary-clinton-emails/Emails.csv'
    df = pd.read_csv(filename)
    print('=> Done.')

    print('Extracting countries and performing sentiment analysis...')
    by_country = process_mails(df)
    print('=> Done.')

    print('Saving data to {}'.format(pickle_file))
    pickle.dump(by_country, open(pickle_file, 'wb'))
    print('=> Done.')

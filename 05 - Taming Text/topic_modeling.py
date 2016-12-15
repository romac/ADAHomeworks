import gensim
from gensim import corpora, models

import pandas as pd
import numpy as np
import nltk
import pickle
import joblib
import re

from nltk.stem.snowball import SnowballStemmer

english_stopwords = nltk.corpus.stopwords.words('english')

local_stopwords = [
    'fyi', 'will', 'call', 'said', 'time', 'want', 'know', 'pm', 'am', 
    'also', 'call', 'know', 'would', 'get', 'time', 'work', 'like', 'today',
    'see', 'morning', 'also', 'back', 'tomorrow', 'meeting', 'think', 'good',
    'want', 'could', 'working', 'well', 'pls', 're', 'fw', 'new', 'thx', 'fwd'
]

stopwords = set(english_stopwords + local_stopwords)

stemmer = SnowballStemmer('english', ignore_stopwords=True)

def tokenize_and_stem(text):

    # First tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # Filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = [token for token in tokens if re.search('^[a-zA-Z]+$', token)]

    # Stem the resulting tokens, filtering out words present in our local stopwords list
    stems = [stemmer.stem(token) for token in filtered_tokens if token not in stopwords]

    # Once again, filter out words present in our local stopwords list
    return [stem for stem in stems if stem not in stopwords and len(stem) > 1]

if __name__ == '__main__':

    print('Loading data...')
    filename='hillary-clinton-emails/Emails.csv'
    df = pd.read_csv(filename)
    print('=> Done.')

    texts = []

    mail_df = df[['ExtractedSubject', 'ExtractedBodyText']].dropna()

    print('Tokenizing...')

    for idx, mail in mail_df.iterrows():
        text = str(mail['ExtractedSubject']) + ' ' + str(mail['ExtractedBodyText'])
        tokens = tokenize_and_stem(text)
        texts.extend(tokens)

    print('=> Done.')

    print('Creating dictionary...')
    dictionary = corpora.Dictionary([texts])
    print('=> Done.')

    print('Creating dictionary...')
    corpus = [dictionary.doc2bow([text]) for text in texts]
    print('=> Done.')

    print('Creating model...')
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=3)
    print('=> Done.')

    joblib.dump(ldamodel, 'gensim_lda_model.pkl')


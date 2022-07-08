import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Training Data
from pathlib import Path
DATAPATH = 'information_retrieval/data/final_transformed_data.csv'
#DATAPATH = Path("data/output/final_transformed_data.csv")

df_original = pd.read_csv(DATAPATH)
#df = df[~df.overview.isna()]

#assign two sentence 
#1 for continent-lang-study combine column
#2 for targetgroup qualification column
df = df_original
df.rename(columns={'continent_language_study_combine':'sentence_1'}, inplace=True)
df.rename(columns={'projects__target_groups_qualifications_combine':'sentence_2'}, inplace=True)


''''''
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

#STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=None, lemmatize=False):
    """
    Lemmatize, tokenize, crop and remove stop words.
    """
    
    tokens = [w for w in word_tokenize(sentence)]
    print(tokens)
    '''
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                        and w not in stopwords)]
    '''
    return tokens

'''    
#Used in Word2vec
def clean_sentences(df):
    """
    Remove irrelavant characters (in new column clean_sentence).
    Lemmatize, tokenize words into list of words (in new column tok_lem_sentence).
    """
    print('Cleaning sentences...')
    #df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['sentence'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=None, lemmatize=True))
    return df
'''

#save tfidf model
import pickle as pk

def save_embedding(model, name):
    with open(f'information_retrieval/model/tfidf_model_{name}.pkl', 'wb') as fn:
    #with open(f'model/tfidf_model_{name}.pkl', 'wb') as fn:
        pk.dump(model, fn)

def save_vectorizer(vectorizer, name):
    with open(f'information_retrieval/model/tfidf_{name}.pkl', 'wb') as fn:
    #with open(f'model/tfidf_{name}.pkl', 'wb') as fn:
        pk.dump(vectorizer, fn)

def train():
    ''''''
    # Adapt stop words
    #token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)

    # Fit TFIDF
    #vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
    vectorizer_1 = TfidfVectorizer(stop_words=None, tokenizer=tokenizer)
    vectorizer_2 = TfidfVectorizer(stop_words=None, tokenizer=tokenizer)
    tfidf_mat_1 = vectorizer_1.fit_transform(df['sentence_1'].values) # -> (num_sentences, num_vocabulary)
    save_embedding(tfidf_mat_1, 'sentence_1')
    save_vectorizer(vectorizer_1, 'vectorizer_1')

    ''''''
    tfidf_mat_2 = vectorizer_2.fit_transform(df['sentence_2'].values) # -> (num_sentences, num_vocabulary)
    save_embedding(tfidf_mat_2, 'sentence_2')
    save_vectorizer(vectorizer_2, 'vectorizer_2')

if __name__ == '__main__':
    train()
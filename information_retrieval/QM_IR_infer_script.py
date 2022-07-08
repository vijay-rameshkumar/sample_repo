#import relevant packages
import pandas as pd
import numpy as np
import pickle as pk
import pycountry_convert as pc
from pathlib import Path
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import statistics

''''''
#Data for matching result index
DATAPATH = 'information_retrieval/data/final_transformed_data.csv'
#DATAPATH = Path("data/output/final_transformed_data.csv") 
df_original = pd.read_csv(DATAPATH)

''''''
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

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    global cos_sim
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index

def get_recommendations(vectorizer, sentence, tfidf_mat):
    
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
    # Embed the query sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
    # Best cosine distance for each token independantly
    print(mat.shape)
    best_index = extract_best_indices(mat, topk=20)
    return best_index

#Load saved model tfidf embedding
def load_embedding(sentence):
    with open(f'information_retrieval/model/tfidf_model_{sentence}.pkl', 'rb') as fn:
        model = pk.load(fn)
    return model

#Load saved model tfidf vectorizer
def load_model(model):
    with open(f'information_retrieval/model/tfidf_{model}.pkl', 'rb') as fn:
        model = pk.load(fn)
    return model

def merge_dict(D1,D2):
    py={**D1,**D2}
    return py

#Main method for Infer
#def IR_infer(req_continent,req_lang,req_study_type,req_subject,req_tg_qualification):
def IR_infer(req_continent_language_study,req_tg_qualification):

    ''''''
    #Load model
    tfidf_embedding_1 = load_embedding('sentence_1')
    tfidf_vectorizer_1 = load_model('vectorizer_1')
    tfidf_embedding_2 = load_embedding('sentence_2')
    tfidf_vectorizer_2 = load_model('vectorizer_2')


    ''''''
    ##sentence_1
    #Calculate match-prediction for sentence_1, for given req parameters
    #query_sentence_1 = req_continent + ' ' + req_lang + ' ' + req_study_type
    query_sentence_1 = req_continent_language_study

    best_tfidf_index_1 = get_recommendations(tfidf_vectorizer_1, query_sentence_1, tfidf_embedding_1)
    matched_df_1 = df_original[['rl']].iloc[best_tfidf_index_1]

    #get scores for matched rl and create key-value pair
    best_cos_sim_list = []
    total_cos_sim_list = cos_sim.tolist()

    for best_index in best_tfidf_index_1.tolist():
        best_cos_sim_list.extend([total_cos_sim_list[best_index]])

    #create rl & match score dict
    rl_score_dict_1 = {}
    i = 0
    for rl in matched_df_1['rl'].tolist():
        if rl in rl_score_dict_1.keys():
            if rl_score_dict_1[rl] < best_cos_sim_list[i]:
                rl_score_dict_1[rl] = best_cos_sim_list[i]
        else:
            rl_score_dict_1[rl] = best_cos_sim_list[i]
        i+=1


    ''''''
    ##sentence_2
    #Calculate match-prediction for sentence_2, for given req parameters
    query_sentence_2 = req_tg_qualification

    best_tfidf_index_2 = get_recommendations(tfidf_vectorizer_2, query_sentence_2, tfidf_embedding_2)
    matched_df_2 = df_original[['rl']].iloc[best_tfidf_index_2]

    #get matched supplier rl and associate score
    best_cos_sim_list = []
    total_cos_sim_list = cos_sim.tolist()

    for best_index in best_tfidf_index_2.tolist():
        best_cos_sim_list.extend([total_cos_sim_list[best_index]])

    #create rl & match score dict
    rl_score_dict_2 = {}
    i = 0
    for rl in matched_df_2['rl'].tolist():
        if rl in rl_score_dict_2.keys():
            if rl_score_dict_2[rl] < best_cos_sim_list[i]:
                rl_score_dict_2[rl] = best_cos_sim_list[i]
        else:
            rl_score_dict_2[rl] = best_cos_sim_list[i]
        i+=1

    ''''''
    #find common supplier keys
    common_suppier_rl = [x for x in rl_score_dict_1 if x in rl_score_dict_2]

    ''''''
    ## Logic for matching and sorting common supplier rl
    #set scaling factor w.r.t 100%
    scale_factor_1 = 1.93
    high_score_1 = 0.35*scale_factor_1
    medium_score_1 = 0.16*scale_factor_1

    scale_factor_2 = 2.75
    high_score_2 = 0.25*scale_factor_2
    medium_score_2 = 0.11*scale_factor_2

    ''''''
    common_suppier_rl_high = {}
    common_suppier_rl_medium_high = {}
    common_suppier_rl_high_medium = {}
    common_suppier_rl_medium_medium = {}
    common_suppier_rl_low_high = {}
    common_suppier_rl_high_low = {}
    threshold = float(0.30)

    for rl_item in common_suppier_rl:
        #print((rl_score_dict_1[rl_item])*scale_factor_1)
        #high & high
        if (float(rl_score_dict_1[rl_item])*scale_factor_1 > high_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 > high_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_high[rl_item] = rl_mean

        #medium & high
        elif (float(rl_score_dict_1[rl_item])*scale_factor_1 < high_score_1) and (float(rl_score_dict_1[rl_item])*scale_factor_1 > medium_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 > high_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_medium_high[rl_item] = rl_mean

        #high & medium
        elif (float(rl_score_dict_1[rl_item])*scale_factor_1 > high_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 < high_score_2) and (float(rl_score_dict_2[rl_item])*scale_factor_2 > medium_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_high_medium[rl_item] = rl_mean

        #medium & medium
        elif (float(rl_score_dict_1[rl_item])*scale_factor_1 < high_score_1) and (float(rl_score_dict_1[rl_item])*scale_factor_1 > medium_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 < high_score_2) and (float(rl_score_dict_2[rl_item])*scale_factor_2 > medium_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_medium_medium[rl_item] = rl_mean

        #low & high       
        elif (float(rl_score_dict_1[rl_item])*scale_factor_1 < medium_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 > high_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_low_high[rl_item] = rl_mean

        #high & low
        elif (float(rl_score_dict_1[rl_item])*scale_factor_1 > high_score_1) and (float(rl_score_dict_2[rl_item])*scale_factor_2 < medium_score_2):
            rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]*scale_factor_1),float(rl_score_dict_2[rl_item]*scale_factor_2)])
            
            if rl_mean > threshold:
                common_suppier_rl_high_low[rl_item] = rl_mean

    common_suppier_rl_high = dict(sorted(common_suppier_rl_high.items(), key=lambda item: item[1], reverse=True))
    common_suppier_rl_medium_high = dict(sorted(common_suppier_rl_medium_high.items(), key=lambda item: item[1], reverse=True))
    common_suppier_rl_high_medium = dict(sorted(common_suppier_rl_high_medium.items(), key=lambda item: item[1], reverse=True))
    common_suppier_rl_medium_medium = dict(sorted(common_suppier_rl_medium_medium.items(), key=lambda item: item[1], reverse=True))
    common_suppier_rl_low_high = dict(sorted(common_suppier_rl_low_high.items(), key=lambda item: item[1], reverse=True))
    common_suppier_rl_high_low = dict(sorted(common_suppier_rl_high_low.items(), key=lambda item: item[1], reverse=True))

    final_supplier_score_dict = common_suppier_rl_high.copy()
    final_supplier_score_dict = merge_dict(final_supplier_score_dict,common_suppier_rl_medium_high)
    final_supplier_score_dict = merge_dict(final_supplier_score_dict,common_suppier_rl_high_medium)
    final_supplier_score_dict = merge_dict(final_supplier_score_dict,common_suppier_rl_medium_medium)
    final_supplier_score_dict = merge_dict(final_supplier_score_dict,common_suppier_rl_low_high)
    final_supplier_score_dict = merge_dict(final_supplier_score_dict,common_suppier_rl_high_low)

    return final_supplier_score_dict

    '''
    #calculate mean score of common supplier
    common_suppier_rl_mean = {}
    threshold = float(0.20)
    for rl_item in common_suppier_rl:
        rl_mean = statistics.mean([float(rl_score_dict_1[rl_item]),float(rl_score_dict_2[rl_item])])
        if rl_mean > threshold:
            common_suppier_rl_mean[rl_item] = rl_mean

    common_suppier_rl_mean_sorted = dict(sorted(common_suppier_rl_mean.items(), key=lambda item: item[1]))
    
    return common_suppier_rl_mean_sorted
    '''

    '''
    #combining 1 & 2 list and dropping duplicate supplier rl values.
    rl_list = []
    rl_list = matched_df_1['rl'].tolist()
    rl_list.extend(matched_df_2['rl'].tolist())
    rl_list_unique = list(set(rl_list))
    rl_list_unique_str = [str(x) for x in rl_list_unique]

    #return rl value
    return rl_list_unique_str
    '''

# ### Libraries
import pandas as pd 
import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs 
from typing import Dict, Text
from tensorflow.keras import layers  
from tensorflow.keras import Model
import tensorflow_ranking as tfr
from tensorflow.keras import optimizers
import pycountry_convert as pc
from scipy.stats import kendalltau
import json
import itertools

import warnings
warnings.filterwarnings('ignore')


# ### Model1 - Ranking Prediction Model (RMSE)

with open('recommender_system/utilities/continent_lang_study_sub_unique_suppliers_subjects_lookup.json', 'r') as openfile:
    json_object = json.load(openfile)

unique_subjects = json_object['unique_subjects']
unique_suppliers = json_object['unique_suppliers']

with open('recommender_system/utilities/layer1_suppliers_lookup.json', 'r') as openfile:
    layer1_suppliers_lookup = json.load(openfile)

with open('recommender_system/utilities/layer2_suppliers_lookup.json', 'r') as openfile:
    layer2_suppliers_lookup = json.load(openfile)

with open('recommender_system/lookup/continent_level_ranking.json', 'r') as openfile:
    layer3_suppliers_lookup = json.load(openfile)

with open('recommender_system/lookup/global_level_ranking.json', 'r') as openfile:
    layer4_suppliers_lookup = json.load(openfile)
l4_suppliers_lookup = [list(i) for i in list(layer4_suppliers_lookup.items())]


##### Weighted Harmonic Mean ##########
def weighted_harmonic_mean(x, y, x_w=0.6, y_w=0.4):
    if x!=0.0 and y != 0.0:
        return 1 / ((x_w/x)+(y_w/y))
    elif x==0.0 and y <= 0.0:
        return 0.09
    elif y<=0:
        return 0
    elif x<=0:
        return 0

####### Ranking Model #################
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 64

    # Compute embeddings for users.
    self.supplier_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_suppliers, mask_token=None),
      tf.keras.layers.Embedding(len(unique_suppliers) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.subject_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_subjects, mask_token=None),
      tf.keras.layers.Embedding(len(unique_subjects) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="leaky_relu"),
      tf.keras.layers.Dense(32, activation="leaky_relu"),
      tf.keras.layers.Dense(16, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    supplier_id, subject_id = inputs

    supplier_embedding = self.supplier_embeddings(supplier_id)
    subject_embedding = self.subject_embeddings(subject_id)

    return self.ratings(tf.concat([supplier_embedding, subject_embedding], axis=1))

############### Recommendation Model #####################
class SupplierRecommender(tfrs.models.Model):
  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["supplier_id"], features["subject_id"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("score")
    rating_predictions = self(features)
    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)

################ Model Constructor ################
model1 = SupplierRecommender()
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model1.load_weights('recommender_system/model/final_model_weights/')


############### Model Predict ##################
def model1_predict(list_of_suppliers, subjects, model=model1):
    preds = []

    for i in list_of_suppliers:
        prediction = model({
            "supplier_id": np.array([i]),
            "subject_id": np.array([subjects])
            }).numpy().tolist()[0][0]
        
        preds.append([i, prediction])
    return preds

################ IR supplier Infer #############
def ir_supplier_infer(supplier_key, panelist_key, suppliers, top_k=10):
    req = top_k

    key1 = supplier_key + "@" + panelist_key.split("@")[0]
    key2 = supplier_key.split("@")[0] + "@" + panelist_key.split("@")[0]
    key3 = supplier_key.split("@")[0]
    
    keys = supplier_key+"@"+panelist_key

    ############# layer 1 ##############
    supp = []
    for i in layer1_suppliers_lookup.keys():
        if i.startswith(key1):
            supp.extend(layer1_suppliers_lookup[i])    
    
    supp = list(set(supp))

    ir_l1_suppliers = list(set(suppliers).intersection(set(supp)))

    model1_pred_order = model1_predict(ir_l1_suppliers, panelist_key)
    suppliers1 = pd.DataFrame(model1_pred_order, columns=['supplier_ref', 'score'])
    suppliers1 = suppliers1.sort_values(by='score', ascending=False).head(top_k)
    suppliers1 = suppliers1[~suppliers1.score.isin([suppliers1.score.mode().values[0]])]

    common_supplier = list(set(suppliers).intersection(set(suppliers1.supplier_ref.unique().tolist())))
    diff_l1 = set(suppliers).difference(set(suppliers1.supplier_ref.unique().tolist()))

    if suppliers == None:
        suppliers1['info'] = 'l1'
    elif suppliers != None:
        suppliers1['info'] = 'IR_l1'
    
    suppliers1['key'] = key1
    suppliers1.drop_duplicates(subset=['supplier_ref'], inplace=True).reset_index(drop=True)
    
    req = req - len(suppliers1)
    final = suppliers1

    # ############## layer 2 ##############
    if req > 0:

        l2 = layer2_suppliers_lookup.get(key2, None)
        supp = set(list(l2.keys()))

        diff_l2 = set(suppliers).difference(set(suppliers1.supplier_ref.unique().tolist()))
        common_supplier = list(set(diff_l2).intersection(supp))

        k2 = [[i, l2[i]] for i in common_supplier]

        if l2 != None:
            suppliers2 = pd.DataFrame(k2, columns=['supplier_ref', 'score'])
            suppliers2 = suppliers2.head(top_k)
            suppliers2['info'] = 'IR_l2'
            suppliers2['key'] = key2
        else:
            suppliers2 = pd.DataFrame(columns=['supplier_ref', 'score'])
        
        suppliers2 = suppliers2.sort_values(by='score', ascending=False)
        final = pd.concat([final, suppliers2])
        final.drop_duplicates(subset='supplier_ref', inplace=True)
        final = final.head(top_k).reset_index(drop=True)
        final = final.head(top_k)
        req = req - len(final)
    
    ############## layer 3 ##############
        if req > 0:
            l3 = layer3_suppliers_lookup.get(key3, None)
            supp = set(list(l3.keys()))

            diff_l3 = set(suppliers).difference(set(final.supplier_ref.unique().tolist()))
            common_supplier = list(set(diff_l3).intersection(supp))

            k3 = [[i, l3[i]] for i in common_supplier]

            if l3 != None:
                suppliers3 = pd.DataFrame(k3, columns=['supplier_ref', 'score'])
                suppliers3['info'] = 'IR_l3'
                suppliers3['key'] = key3
                
            else:
                suppliers3 = pd.DataFrame(columns=['supplier_ref', 'score'])
            
            suppliers3 = suppliers3.sort_values(by='score', ascending=False).head(top_k).reset_index(drop=True)
            final = pd.concat([final, suppliers3])
            final.drop_duplicates(subset='supplier_ref', inplace=True)
            
            final = final.head(top_k)
            req = req - len(final)

    # # ############# layer 4 ###############
            if req > 0:

                diff_l4 = set(list(layer4_suppliers_lookup.keys())).difference(suppliers)
                common_supplier = diff_l4.intersection(set(list(layer4_suppliers_lookup.keys())))

                k4 = [[i, layer4_suppliers_lookup[i]] for i in common_supplier]

                suppliers4 = pd.DataFrame(k4, columns=['supplier_ref', 'score'])
                suppliers4['info'] = 'IR_l4'
                suppliers4['key'] = 'global'
                suppliers4 = suppliers4.sort_values(by='score', ascending=False).reset_index(drop=True)

                final = pd.concat([final, suppliers4])
                final.drop_duplicates(subset='supplier_ref', inplace=True)
                final = final.head(top_k)

    return final.reset_index(drop=True)

############### Recsys - Inference #################
############### Inference #################
def recsys_infer(supplier_key, panelist_key, top_k=10, suppliers=None):
    req = top_k

    key1 = supplier_key + "@" + panelist_key.split("@")[0]
    key2 = supplier_key.split("@")[0] + "@" + panelist_key.split("@")[0]
    key3 = supplier_key.split("@")[0]
    
    keys = supplier_key +"@"+ panelist_key

    ############# layer 1 ##############
    supp = []
    if suppliers == None:
        for i in layer1_suppliers_lookup.keys():
            if i.startswith(key1):
                supp.extend(layer1_suppliers_lookup[i])

        # for i in unique_suppliers:
        #     if i.startswith(key1):
        #         supp.extend(layer1_suppliers_lookup[i])
        
    elif suppliers:
        supp.extend(suppliers)
    
    supp = list(set(supp))
    #print(supp)

    model1_pred_order = model1_predict(supp, panelist_key)
    # model1_pred_order.sort(key=lambda x:x[1], reverse=True)
    suppliers1 = pd.DataFrame(model1_pred_order, columns=['supplier_ref', 'score'])
    suppliers1 = suppliers1.sort_values(by='score', ascending=False).head(top_k)

    if suppliers == None:
        suppliers1['info'] = 'l1'
    elif suppliers != None:
        suppliers1['info'] = 'IR'
    
    suppliers1['key'] = key1
    suppliers1.drop_duplicates(subset=['supplier_ref'], inplace=True)
    
    req = len(suppliers1)
    final = suppliers1

    ############## layer 2 ##############
    if req < top_k:
        l2 = layer2_suppliers_lookup.get(key2, None)

        if l2 != None:
            suppliers2 = pd.DataFrame(l2.items(), columns=['supplier_ref', 'score'])
            suppliers2 = suppliers2.head(top_k)
            suppliers2['info'] = 'l2'
            suppliers2['key'] = key2
        else:
            suppliers2 = pd.DataFrame()
        
        suppliers2 = suppliers2.sort_values(by='score', ascending=False)
        final = pd.concat([final, suppliers2])
        final.drop_duplicates(subset='supplier_ref', inplace=True)
        final = final.head(top_k)
        req = req - len(final)
    
    # ############## layer 3 ##############
        if req <= 0:
            l3 = layer3_suppliers_lookup.get(key3, None)

            if l3 != None:
                suppliers3 = pd.DataFrame(l3.items(), columns=['supplier_ref', 'score'])
                suppliers3['info'] = 'l3'
                suppliers3['key'] = key3
                
            else:
                suppliers3 = pd.DataFrame()
            
            suppliers3 = suppliers3.sort_values(by='score', ascending=False)
            final = pd.concat([final, suppliers3])
            final.drop_duplicates(subset='supplier_ref', inplace=True)
            final = final.head(top_k)
            req = req - len(final)

    # ############# layer 4 ###############
            if req <= 0:
                suppliers4 = pd.DataFrame(l4_suppliers_lookup[:top_k], columns=['supplier_ref', 'score'])
                suppliers4['info'] = 'l4'
                suppliers4['key'] = 'global'
                suppliers4 = suppliers4.sort_values(by='score', ascending=False)
                final = pd.concat([final, suppliers4])
                final.drop_duplicates(subset='supplier_ref', inplace=True)
                final = final.head(top_k)

    return final.reset_index(drop=True)

if __name__ == '__main__':
    actual_layer1 = pd.read_csv('recommender_system/data/adj_matrix/01_adjacency_continent_lang_study_subject_weighted_suppliers_average_profit.csv')
    suppliers = actual_layer1.suppliers_info.values.tolist()
    actual_layer1 = actual_layer1.set_index('suppliers_info')
    actual_layer1 = actual_layer1.sort_index()
    actual_layer1 = actual_layer1.stack().reset_index()
    actual_layer1.columns = ['suppliers__ref', 'projects__study_types_subject_ids', 'positive_score']
    actual_layer1['suppliers__ref'] = actual_layer1['suppliers__ref'].astype('str')
    actual_layer1['positive_score'] = actual_layer1['positive_score'].astype('float32')
    actual_layer1 = actual_layer1[actual_layer1.positive_score != 0.00]

    cont_lang = actual_layer1.suppliers__ref.apply(lambda x: "@".join(x.split('@')[:-1])).unique().tolist()
    study_sub = actual_layer1.projects__study_types_subject_ids.unique().tolist()
    total = [cont_lang, study_sub]
    combinations = [p for p in itertools.product(*total)]

    for index, comb in enumerate(combinations):
        final1 = recsys_infer(comb[0], comb[1], top_k=10, suppliers=None)
        break

    print(final1)

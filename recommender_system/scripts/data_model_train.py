#### Importing Libraries ##########
import os
import pprint
import tempfile

from typing import Dict, Text
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

#### Read Data ####################

positive_samples = pd.read_csv('recommender_system/data/adj_matrix/01_adjacency_continent_lang_study_subject_weighted_suppliers_average_profit.csv')
suppliers = positive_samples.suppliers_info.values.tolist()
positive_samples = positive_samples.set_index('suppliers_info')
positive_samples = positive_samples.sort_index()
positive_samples = positive_samples.stack().reset_index()

positive_samples.columns = ['suppliers__ref', 'projects__study_types_subject_ids', 'positive_score']

positive_samples['suppliers__ref'] = positive_samples['suppliers__ref'].astype('str')
positive_samples['positive_score'] = positive_samples['positive_score'].astype('float32')
# positive_samples = positive_samples[positive_samples.positive_score != 0.0]
# test = pd.pivot(positive_samples, index=['suppliers__ref'], columns='projects__study_types_subject_ids', values='positive_score')

###### Prepare the Data ###########

training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(positive_samples['suppliers__ref'].values, tf.string),
            tf.cast(positive_samples['projects__study_types_subject_ids'].values, tf.string),
            tf.cast(positive_samples['positive_score'].values, tf.float32)
        )
    )
)

ratings = training_dataset.map(lambda x,y,z: {
    "subject_id": y,
    "supplier_id": x,
    "score": z,
})

subjects = ratings.map(lambda x:x['subject_id'])

unique_subjects = positive_samples['projects__study_types_subject_ids'].unique()
unique_suppliers = positive_samples['suppliers__ref'].unique()

######## train-test shuffle ############
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

######## Model Architecture ############

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

######### Model Train ##############
epoch=2000

cached_train = train.shuffle(100_000).batch(10).cache()
cached_test = test.batch(2).cache()

model = SupplierRecommender()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

callback = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', patience=50, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('model/new_recsys_model_weights', save_best_only=True, monitor='root_mean_squared_error', mode='min')

history = model.fit(cached_train, epochs=epoch, callbacks=[callback, checkpoint])

######### save model to disk ##########
model.save_weights('recommender_system/model/final_model_weights/')

########## Plot-Save History ##################
def plot(x, y, xlabel, ylabel, name):
  plt.figure(figsize=(15,7)) 
  plt.plot(x, y)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(f'recommender_system/output/{name}.png')

plot(list(range(1, len(history.history['root_mean_squared_error'])+1)), history.history['root_mean_squared_error'], 'epochs', 'rmse', 'rmse_history')
plot(list(range(1, len(history.history['total_loss'])+1)), history.history['total_loss'], 'epochs', 'total_loss', 'loss_history')
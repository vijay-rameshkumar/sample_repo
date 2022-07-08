# %%
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from typing import Dict, Text
import pycountry_convert as pc
import pycountry
import json
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def weighted_harmonic_mean(x, y, x_w=0.6, y_w=0.4):
    if x!=0.0 and y != 0.0:
        return 1 / ((x_w/x)+(y_w/y))
    elif x==0.0 and y <= 0.0:
        return 0.09
    elif y<=0:
        return 0
    elif x<=0:
        return 0

# ## Adjacency Matrix 1
df1 = pd.read_csv('recommender_system/data/01_continent_lang_study_subject_weighted_suppliers_average_profit.csv')

supp = df1[['continents', 'sample_pulls__language', 'suppliers__ref']].values.tolist()
sub = df1[['projects__study_types_ids', 'projects__study_types_subject_ids']].values.tolist()

df1['suppliers_info'] = pd.Series(["@".join([str(k) for k in j]) for j in supp])
df1['study_subject'] = pd.Series(["@".join(j) for j in sub])

df1['score'] = list(map(weighted_harmonic_mean, df1.median_profit_per_respondent.values.tolist(), df1.weighted_suppliers_median_complete_ratio.values.tolist()))

df1 = pd.pivot(df1, index='suppliers_info', columns='study_subject', values='score').fillna(0)

if os.path.exists('recommender_system/data/adj_matrix'):
    pass
else:
    os.mkdir('recommender_system/data/adj_matrix')
df1.reset_index().to_csv('recommender_system/data/adj_matrix/01_adjacency_continent_lang_study_subject_weighted_suppliers_average_profit.csv', index=False)


# ## Adjacency Matrix 2
df2 = pd.read_csv('recommender_system/data/02_continent_study_weighted_suppliers_average_profit.csv')

supp = df2[['continents', 'suppliers__ref']].values.tolist()
sub = df2[['projects__study_types_ids']].values.tolist()

df2['suppliers_info'] = pd.Series(["@".join([str(k) for k in j]) for j in supp])
df2['study'] = df2['projects__study_types_ids']

df2['score'] = list(map(weighted_harmonic_mean, df2.median_profit_per_respondent.values.tolist(), df2.weighted_suppliers_median_complete_ratio.values.tolist()))

df2 = pd.pivot(df2, index='suppliers_info', columns='study', values='score').fillna(0)

if os.path.exists('recommender_system/data/adj_matrix'):
    pass
else:
    os.mkdir('recommender_system/data/adj_matrix')

df2.reset_index().to_csv('recommender_system/data/adj_matrix/02_continent_study_weighted_suppliers_average_profit.csv', index=False)


# ## Adjacency Matrix 3 - Weighted Harmonic Mean

df1 = pd.read_csv('recommender_system/data/01_continent_lang_study_subject_weighted_suppliers_average_profit.csv')

supp = df1[['continents', 'sample_pulls__language', 'suppliers__ref']].values.tolist()
sub = df1[['projects__study_types_ids', 'projects__study_types_subject_ids']].values.tolist()

df1['suppliers_info'] = pd.Series(["@".join([str(k) for k in j]) for j in supp])
df1['study_subject'] = pd.Series(["@".join(j) for j in sub])

scores1 = df1['median_profit_per_respondent'].values.tolist()
scores2 = df1['weighted_suppliers_median_complete_ratio'].values.tolist()

df1['final_score'] = list(map(weighted_harmonic_mean, scores1,scores2))

df1 = pd.pivot(df1, index='suppliers_info', columns='study_subject', values='final_score').fillna(0)

if os.path.exists('recommender_system/data/adj_matrix'):
    pass
else:
    os.mkdir('recommender_system/data/adj_matrix')

df1.reset_index().to_csv('recommender_system/data/adj_matrix/03_adjacency_continent_lang_study_subject_weighted_suppliers_average_profit.csv', index=False)


# ## Continent Level Lookup
df3 = pd.read_csv('recommender_system/data/03_continent_weighted_suppliers_average_profit.csv')[['continents', 'median_profit_per_respondent', 'weighted_suppliers_median_complete_ratio', 'suppliers__ref']]
df3['suppliers__ref'] = df3['suppliers__ref'].astype('str')
df3['score'] = list(map(weighted_harmonic_mean, df3['median_profit_per_respondent'].values.tolist(), df3['weighted_suppliers_median_complete_ratio'].values.tolist()))
df3 = df3.sort_values(by=['score'], ascending=False)

continent_lookup = {}

for i, j in df3.groupby(['continents']):
    continent_lookup[i] = dict(j[['suppliers__ref', 'score']].values.tolist())

obj = json.dumps(continent_lookup, indent=4)

with open('recommender_system/lookup/continent_level_ranking.json', 'w') as f:
    f.write(obj)

# ## Global Lookup
df4 = pd.read_csv('recommender_system/data/04_Global_weighted_suppliers_average_profit.csv')[['median_profit_per_respondent', 'weighted_suppliers_median_complete_ratio', 'suppliers__ref']]

df4['suppliers__ref'] = df4['suppliers__ref'].astype('str')
df4['score'] = list(map(weighted_harmonic_mean, df4['median_profit_per_respondent'].values.tolist(), df4['weighted_suppliers_median_complete_ratio'].values.tolist()))
df4 = df4.sort_values(by=['score'], ascending=[False])

obj = json.dumps(dict(df4[['suppliers__ref', 'score']].values.tolist()), indent=4)

with open('recommender_system/lookup/global_level_ranking.json', 'w') as f:
    f.write(obj)

# ## Important_lookups


# subjects_lookup = {'study' : df2['study'].str.strip().unique().tolist()}

# json_obj = json.dumps(subjects_lookup, indent=4)
# with open('utils_v2/continent_study_lookup.json', 'w') as f:
#     f.write(json_obj)

# supplier_lookup = {}

# for con in df2[['continents', 'suppliers__ref']].drop_duplicates().continents.unique():
#     supplier_lookup[con] = df2[df2.continents == con].suppliers_info.unique().tolist()

# json_obj = json.dumps(supplier_lookup, indent=4)
# json_obj

# with open('utils_v2/continent_supplier_lookup.json', 'w') as f:
#     f.write(json_obj)

unique_suppliers_subjects_lookup = {'unique_suppliers' : df1.reset_index().suppliers_info.unique().tolist(), 'unique_subjects':list(df1.reset_index().columns[1:])}

json_obj = json.dumps(unique_suppliers_subjects_lookup, indent=4)

if os.path.exists('recommender_system/data/utils_v2'):
    pass
else:
    os.mkdir('recommender_system/data/utils_v2')

with open('recommender_system/data/utils_v2/continent_lang_study_sub_unique_suppliers_subjects_lookup.json', 'w') as f:
    f.write(json_obj)

unique_suppliers_subjects_lookup = {'unique_subjects' : df2.reset_index().suppliers_info.unique().tolist(), 'unique_suppliers':list(df2.columns[1:])}

json_obj = json.dumps(unique_suppliers_subjects_lookup, indent=4)

with open('recommender_system/data/utils_v2/continent_study_unique_suppliers_subjects_lookup.json', 'w') as f:
    f.write(json_obj)

# ## Extra Lookup to simplify the process
df1 = pd.read_csv('recommender_system/data/01_continent_lang_study_subject_weighted_suppliers_average_profit.csv')
df1['score'] = list(map(weighted_harmonic_mean, df1.median_profit_per_respondent.values.tolist(), df1.weighted_suppliers_median_complete_ratio.values.tolist()))
df1['key'] = df1['continents']+"@"+df1['sample_pulls__language']+"@"+ df1['projects__study_types_ids']

lookup = df1[['key', 'suppliers__ref', 'score']].sort_values(by = ['key', 'score'], ascending =[True, False])[['key', 'suppliers__ref']]
lookup['suppliers__ref'] = lookup['suppliers__ref'].astype('str')
lookup = lookup.drop_duplicates(subset=['key', 'suppliers__ref'])
lookup = lookup.groupby(['key']).suppliers__ref.agg(",".join).reset_index(name='suppliers__ref')
lookup['suppliers__ref'] = lookup.suppliers__ref.apply(lambda x: x.split(','))

json_obj = json.dumps(dict(lookup.values.tolist()), indent=4)

with open('recommender_system/data/utils_v2/layer1_suppliers_lookup.json', 'w') as f:
    f.write(json_obj)

df2 = pd.read_csv('recommender_system/data/02_continent_study_weighted_suppliers_average_profit.csv')
df2['score'] = list(map(weighted_harmonic_mean, df2.median_profit_per_respondent.values.tolist(), df2.weighted_suppliers_median_complete_ratio.values.tolist()))
df2['suppliers__ref'] = df2['suppliers__ref'] .astype('str')

df2['key'] = df2['continents'] + "@" + df2['projects__study_types_ids']
lookup = df2[['key', 'suppliers__ref', 'score']].sort_values(by=['key', 'score'], ascending=[True,False]).reset_index(drop=True)
lookup

layer2_lookup = {}

for i, j in lookup.groupby(['key']):
    layer2_lookup[i] = dict(j[['suppliers__ref', 'score']].values.tolist())


json_obj = json.dumps(layer2_lookup, indent=4)
with open('recommender_system/data/utils_v2/layer2_suppliers_lookup.json', 'w') as f:
    f.write(json_obj)

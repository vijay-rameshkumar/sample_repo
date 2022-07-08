# @author <a href="mailto:vijay.rameshkumar@valuelabs.com">Vijay Rameshkumar</a>
# @created date <a href="mailto:date@24/06/2022">Created Date</a>

############## import packages ####################
import json
from flask import Flask
from flask import Flask, request, jsonify
from recommender_system.inference import  recsys_infer
from recommender_system.inference import  ir_supplier_infer, weighted_harmonic_mean
from information_retrieval.QM_IR_infer_script import IR_infer as ir_infer
from information_retrieval import QM_IR_train_script
from infer_data_transform import main_transform

import pandas as pd
import numpy as np
import argparse as args
import pycountry_convert as pc
import pycountry
from data_transformation import study_groups, save_to_csv

########### methods ####################
get_continent = lambda x: pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(x))).lower().replace(" ", "-")

########### Flask API #################################
app = Flask(__name__)

@app.route('/recommend_suppliers', methods=['POST'])
def query():
    req_data = request.get_json()
    study_groups = req_data.get('study_groups')
    tg_groups = req_data.get('target_group')
    language = req_data.get('language')
    country = req_data.get('country')
    top_k = int(req_data.get('top_k'))

    print(country, language, study_groups, tg_groups)


    # df = pd.read_csv('data\source\inference_data.csv').head(1)

    df = pd.DataFrame([country.upper(), language, study_groups, tg_groups]).T
    df.columns = ['sample_pulls__country', 'sample_pulls__language', 'projects__study_types', 'projects__target_groups']
    
    for i in df.columns:
        df[i] = df[i].astype('str')

    transformed = main_transform(df, is_train=False)

    supplier_list = []
    panelist_for = transformed.projects__study_types_ids.values.tolist()[0].lower().replace(" ", "_") + "@" + transformed.projects__study_types_subject_ids.values.tolist()[0].lower().replace(" ", "_")
    supplier_for = transformed.continent.values.tolist()[0].lower().replace(" ", "-")+ "@" + transformed.sample_pulls__language.values.tolist()[0].lower()

    ############# IR sample params #################
    #INFERENCE parameters 
    req_continent_language_study = transformed.continent_language_study_combine.values.tolist()[0]
    req_tg_qualification = transformed.projects__target_groups_qualifications_combine.astype('str').values.tolist()[0]

    ###### IR infer ############################
    # supplier_list = ir_infer(req_continent,req_lang,req_study_type,req_subject,req_tg_qualification)
    supplier_list = ir_infer(req_continent_language_study,req_tg_qualification)

    # # ###### recsys_infer #########################
    ir_suppliers = [str(i) for i in list(supplier_list.keys())]

    if len(ir_suppliers) != 0:
        if top_k > len(ir_suppliers):
            recommendations = ir_supplier_infer(supplier_for, panelist_for, suppliers=ir_suppliers, top_k=top_k)
            recommendations['ir_score'] = recommendations.supplier_ref.apply(lambda x: supplier_list.get(int(x), np.nan))
            recommendations['final_score'] = list(map(lambda x, y: (x+y)/2, recommendations.score.values.tolist(), recommendations.ir_score.values.tolist()))
            
            recommendations = recommendations[['supplier_ref', 'final_score', 'info', 'key']].dropna().reset_index(drop=True)

            recsys_rec_results = recsys_infer(supplier_for, panelist_for, suppliers=None, top_k=top_k)
            recsys_rec_results = recsys_rec_results[~recsys_rec_results.supplier_ref.isin(recommendations.supplier_ref.unique().tolist())]
            recsys_rec_results.columns = ['supplier_ref', 'final_score', 'info', 'key']

            recommendations = pd.concat([recommendations, recsys_rec_results]).reset_index(drop=True).head(top_k)

    else:
        recommendations = recsys_infer(supplier_for, panelist_for, suppliers=ir_suppliers, top_k=top_k).head(top_k)
    
    
    return jsonify(recommendations.to_dict())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True, )
    # _ = query()
# @author <a href="mailto:vijay.rameshkumar@valuelabs.com">Vijay Rameshkumar</a>
# @created date <a href="mailto:date@24/06/2022">Created Date</a>

############## import packages ####################
from recommender_system.inference import  recsys_infer
from recommender_system.inference import  ir_supplier_infer, weighted_harmonic_mean
from information_retrieval.QM_IR_infer_script import IR_infer as ir_infer
from information_retrieval import QM_IR_train_script

import pandas as pd
import numpy as np
import argparse as args
import pycountry_convert as pc
import pycountry
import data_transformation as dt
from data_transformation import study_groups, save_to_csv

aws_root_path = '/home/ec2-user/SageMaker/supplier-recommendation-engine'
data_transform_path = 'data/source/inference_data.csv'
is_train = False
dt.main_transform(data_transform_path, is_train)


############# Initialize the Parser ################
parser = args.ArgumentParser(description ='Processing your queries...')

parser.add_argument('-c', '--country', default='can', help='panelist country', type=str)
parser.add_argument('-l', '--language', default='eng', help='panelist language', type=str)
parser.add_argument('-t', '--study_type', default='b2b', help='panelist study', type=str)
parser.add_argument('-s', '--subject', default='entertainment', help='panelist subject specialisation', type=str)
#parser.add_argument('-p', '--data_path', default='data/output/final_transformed_data.csv', help='source data path', type=str)
parser.add_argument('-irp', '--ir_data_path', default='information_retrieval/data/inference_transformed_data.csv', help='ir source data path', type=str)
parser.add_argument('-k', '--top_k', default=20, help='how many top_k supplier you want', type=int)

args = parser.parse_args()

############ Read Data #######################
ir_query = pd.read_csv(args.ir_data_path, nrows=1)[['continent', 'sample_pulls__language', 'projects__study_types_ids', 'projects__study_types_subject_ids','continent_language_study_combine','projects__target_groups_qualifications_combine']]

query = ir_query

# pd.read_csv(args.ir_data_path, nrows=1)[['continent', 'sample_pulls__language', 'projects__study_types_ids', 'projects__study_types_subject_ids']]

############ region mapping ##################
get_continent = lambda x: pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(x))).lower().replace(" ", "-")
#print(query.head())

############# variables     #######################
supplier_list = []
panelist_for = query.projects__study_types_ids.values.tolist()[0].lower().replace(" ", "_") + "@" + query.projects__study_types_subject_ids.values.tolist()[0].lower().replace(" ", "_")
supplier_for = query.continent.values.tolist()[0].lower().replace(" ", "-")+ "@" + query.sample_pulls__language.values.tolist()[0].lower()

############# IR sample params #################
#INFERENCE parameters 
req_continent_language_study = ir_query.continent_language_study_combine.values.tolist()[0]
req_tg_qualification = ir_query.projects__target_groups_qualifications_combine.values.tolist()[0]

###### IR infer ############################
#supplier_list = ir_infer(req_continent,req_lang,req_study_type,req_subject,req_tg_qualification)
supplier_list = ir_infer(req_continent_language_study,req_tg_qualification)

# ###### recsys_infer #########################
ir_suppliers = [str(i) for i in list(supplier_list.keys())]

if len(ir_suppliers) != 0:
    if args.top_k > len(ir_suppliers):
        recommendations = ir_supplier_infer(supplier_for, panelist_for, suppliers=ir_suppliers, top_k=args.top_k)
        recommendations['ir_score'] = recommendations.supplier_ref.apply(lambda x: supplier_list.get(int(x), np.nan))
        recommendations['final_score'] = list(map(lambda x, y: (x+y)/2, recommendations.score.values.tolist(), recommendations.ir_score.values.tolist()))
        
        recommendations = recommendations[['supplier_ref', 'final_score', 'info', 'key']].dropna().reset_index(drop=True)

        recsys_rec_results = recsys_infer(supplier_for, panelist_for, suppliers=None, top_k=args.top_k)
        recsys_rec_results = recsys_rec_results[~recsys_rec_results.supplier_ref.isin(recommendations.supplier_ref.unique().tolist())]
        recsys_rec_results.columns = ['supplier_ref', 'final_score', 'info', 'key']

        recommendations = pd.concat([recommendations, recsys_rec_results]).reset_index(drop=True).head(args.top_k)

else:
    recommendations = recsys_infer(supplier_for, panelist_for, suppliers=ir_suppliers, top_k=args.top_k).head(args.top_k)

recommendations.to_csv('data/output/final_recommendation.csv', index=False)
print("Supplier Recommendation Process Completed and stored in data/output/final_recommendation.csv")
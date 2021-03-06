from collections import ChainMap
from queue import Empty
import pandas as pd
import ast 
import pycountry_convert as pc

input_file = ""
df = ""
cn = 'projects__study_types'
#input_file = 'data/source/full_ml_training_data_2022-06-23_1655946006563/full_ml_training_data_2022-06-23_1655946006563.csv'

def unique_filter():
    global df

    print(f'dropping duplicates project-suppplier wise...')
    df = df.drop_duplicates(subset=['projects__id','suppliers__id'])

def map_continent():
    global df

    print(f'mapping country to continent...')
    df['continent'] = df.apply(lambda x: pc.convert_continent_code_to_continent_name(pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(x['sample_pulls__country'].upper()))), axis=1)

def combine_lang_study():
    global df

    print(f'combining language and study columns...')
    df['language_study_combine'] = df['sample_pulls__language'].map(str) + ' ' + df['projects__study_types_ids'].map(str)

def combine_continent_lang_study():
    global df

    print(f'combining continent-language-study columns...')
    df['continent_language_study_combine'] = df['continent'].str.replace(" ","") + ' ' + df['sample_pulls__language'].map(str) + ' ' + df['projects__study_types_ids'].map(str)


def map_study_id(x):
    a = {
        "40165df2-2e90-421b-81a7-92d6de784eeb": "B2B",
        "40bfbdc8-e328-44a2-8d99-91779ee7af52": "Community",
        "58ce16af-4615-4c15-b50a-bc3b9f8782af": "Consumer Study",
        "a452991e-25f9-4b8d-a988-780185bb79dc": "Healthcare",
        "0ca996b8-395e-4c9e-966e-017b73fa3c10": "Music Study",
        "7ad4fa71-ef64-46ac-8b50-e1430c8a7b07": "Product Testing",
        "b6d2d721-4017-4289-b869-7269a3339ebd": "Recruit"
    }

    return a[x]

def map_subject_id(lst):
    b = {
        "d13e78ef-48ed-46ea-9aec-12950a164c84":"banking_financial",
        "35b79ac1-9b3a-4752-b58b-aa866b098090":"biotech_pharmaceutical",
        "0c325fe2-6266-4c37-b5be-bc6328197a73":"corporate_travel",
        "859d0917-e6d1-4e02-9f89-7eda7a874975":"fulfillment",
        "79ef05f1-4dac-45f1-814e-a4eeb8155743":"hr",
        "bc5b9962-b316-45eb-998c-1c77d87e7fe0":"it_decision_maker",
        "d2d28831-b843-469c-82f7-33f4c0ba9e25":"legal",
        "c9f8f3a0-e399-4ecf-a0f1-566aef415723":"marketing_advertising",
        "6eeaf2c0-4a88-4c66-b088-8b6318d23b0d":"office_supplies",
        "53aed679-240e-462f-8951-bbd25ce2b9af":"operations",
        "bdeef761-bdae-4f20-9ab9-87eca2fab072":"real_estate_brokers_agents",
        "006c87f4-9287-4a21-85f1-62e0fcb5bfd2":"sales",
        "2bc5a048-0220-4789-a538-ab1959cb9546":"security",
        "1e01718c-d581-48f3-853f-695527bd2ccc":"service",
        "c1815203-0791-446a-bf88-81e1f624f18a":"software_hardware",
        "73d13a7b-4a67-43e8-80b1-0de01a06f555":"technology",
        "1f668cbd-197c-4caf-80b1-fc4b23cc6fc3":"transportation_logistics",
        "f3c73d6a-464e-4adf-a6fe-f66afc3db11f":"other",
        "8ce2e5d3-e5d4-494e-b406-5d53e8e7dbc3":"apparel_fashion ",
        "fc86e9da-c494-46c1-8477-d14db7f06272":"automotive_vehicle ",
        "e102c50c-e18c-45cb-bd7c-0a7b28e8e9ea":"bulletin_board_diary ",
        "e7d01b2b-2dab-4954-8864-e6ff621f844f":"commuting_frequent_travelers ",
        "27ee89ea-2e35-4be6-9210-be379bd5bd46":"electronics_large_appliance",
        "798e6098-76b7-4cf0-8f2f-5bc15167709c":"gaming",
        "51af3b9e-1d2d-4472-928e-ded86323f756":"kitchen_small_appliance",
        "e1a115c4-c711-4389-85f1-5146cbc590ef":"long_term",
        "a7718314-43dc-4683-a2bc-ec5b5801169a":"mattress_bedding ",
        "3aaa98f7-f8e0-4beb-adc9-4d468f243cb0":"mental_health_behavioral",
        "d294a03e-f79c-4ccb-9948-e143758600bd":"parenting_childcare",
        "115a186b-c926-4c7b-9d6f-8fb364e84627":"recruitment_panel",
        "ba7ad211-8661-4243-b0af-251c11fa314a":"shop_along_dining",
        "a9f57ded-68b5-470d-a137-e3ece282f3d5":"short_term",
        "fd5c7eef-2ab1-48f0-9156-3a94d2d63c58":"other",
        "f1623cdf-c1e7-4188-805a-e959ccb80e90":"automotive_transportation",
        "e550107c-28bf-438d-a7b7-587ebc0ff122":"community_insights ",
        "f45fe241-110c-44b5-a9f8-6cac0ebd68bf":"entertainment",
        "6a3517b9-b93d-4f74-9cfc-18c31ecfd28e":"finance_legal_insurance",
        "90d20f7c-da44-4c70-b736-e24e2aeae28a":"food_beverage",
        "5e7188c7-d4c1-426b-8c09-e3e2d33fc4b5":"grooming_cosmetics",
        "a337fdf6-1dc3-4418-9091-bbd3247a5883":"household",
        "bc193f05-1771-4544-b72e-a2e99c1a1823":"news_print_media",
        "8364dc15-8794-4289-a821-e5115187e579":"nutrition_wellness_fitness",
        "825aaab6-7691-4238-b132-0b3a9e6db01b":"pets_animal",
        "8541642e-2020-4810-aeca-555b826a6e3b":"political",
        "a7e9a929-2565-4b90-bc4a-66f757b8f584":"political_civic_services",
        "4830f8e9-5c9b-4492-8e12-e909f8f12f22":"print_social_media",
        "8a6d75b5-7609-4be1-aa4e-37ed2a1870f2":"realestate_construction",
        "a1e5507e-dc92-4a20-91e0-538167a68b03":"shopping_ecommerce ",
        "9421c5f2-1ac7-4000-8b9a-c7d0afe9c9f0":"social_media",
        "61e25b42-46dd-41c1-b28a-2e3ed46db835":"sports_gambling",
        "97a74f67-2f72-466b-8584-8830fe000d75":"technology",
        "a62d7f91-59ed-45d8-b0ac-849ea56d911e":"tobacco_vaping ",
        "e7b86ff7-090b-4f25-b1f4-e0ac94c6d131":"travel",
        "a6d1da3c-a58e-4b7f-9a59-a40e7ea7a92b":"video_games_games_toys",
        "71927908-032c-4eb0-b7aa-4f37ee36c337":"other",
        "c3102234-f46c-4df8-8ee5-e4a1857484dc":"patient",
        "892c04c1-5882-49e0-8923-9ad0b027f67f":"pharmaceutical",
        "2fb3b064-49cc-4ac8-bb38-521352442988":"professional",
        "eeb127d9-ef6a-4829-b06f-6d7ca9b6809c":"other",
        "3a263828-d4c2-4283-8747-064944213cde":"long_form",
        "d4c6299b-19aa-42ce-821e-18ba9f4c7d74":"short",
        "aacec342-f98b-4e50-81bf-7d79b602f639":"other",
        "a0c26f95-1f15-4a26-93a0-1ebccd342fde":"apparel_fashion",
        "6aa0c31f-35bc-409a-bb0c-3bc2a00c1181":"appliance_home_improvement",
        "107af06a-242b-41f5-8ce3-c12db0f3e7b3":"app_online",
        "d780fe64-26ad-4d17-9fac-a37c751e1c08":"automotive_vehicle",
        "f0d2de5d-2233-4374-9b06-c25d0d720c23":"beauty_self_care",
        "91e53495-b403-45bf-ac35-327998e1f79b":"electronics",
        "c9e01dbd-e64d-4552-ae63-210156e49fab":"entertainment_media",
        "2df8da96-7905-4353-86b1-94f6726694f1":"food_beverage",
        "e9bb90fa-c4bb-47ea-89c9-dfb2ac79e3b2":"games_toys",
        "1f2b5133-0155-4651-899b-7890e29ad892":"household_home_care",
        "dc07d54f-deac-4505-919a-f2e307822ec8":"parenting_childcare",
        "e09cfe0f-48a1-40cb-bd41-64ce17e894b6":"pets_animal",
        "b0d3230f-bb2c-47a5-90df-7525d74aa8a4":"tobacco_vaping ",
        "4753d06b-e9e2-41c9-adf4-d91ed43266f9":"other",
        "4e989a79-5e86-464b-9683-ddab15af5de8":"double_opt_in",
        "c51a81e5-7e8c-4937-9b04-3d280151f0a5":"single_opt_in",
        "bef3c165-34d3-494a-8d3e-bff90a5d1571":"other",
    }

    return [b[x] for x in lst]

def map_qual(lst):
    c = {
        "a43cd8e0-426d-465d-a5fd-6a3108bf1211":"AGE_AND_GENDER_OF_CHILD",
        "78e10894-3939-4df6-b2e6-48aea84778c9":"STANDARD_VOTE",
        "4b26ffc7-f3f9-4575-b050-bfd359cdef12":"STANDARD_PETS",
        "3976380f-1287-4a41-a7c9-705b443fb6c4":"GENDER",
        "9dc46ac0-83a8-4c36-a98f-fef481be9cbb":"STANDARD_PRIMARY_DECISION_MAKER",
        "83613519-15b7-4aa5-9306-b197aa7850d3":"STANDARD_TOTAL_HOUSEHOLD",
        "cd0789fe-8690-4570-9c5f-18b0e3a391a9":"STANDARD_DIAGNOSED_AILMENTS_I",
        "11dbef0c-1e9f-4414-9e5a-8c0e94df9820":"STANDARD_EMPLOYMENT",
        "37410804-6835-465a-a23f-505af025b4ae":"HISPANIC",
        "a76ccdfb-ba3a-4d2c-8dd3-b195777ca480":"STANDARD_DIAGNOSED_AILMENTS_II",
        "0b704231-66f2-4c73-9b46-98bae3842d67":"STANDARD_HOUSEHOLD_TYPE",
        "f2551596-cd3e-4922-bb69-82d5e04c8d3c":"STANDARD_AUTO_BRANDS",
        "2b20f529-64ca-4d44-b23d-01238b886794":"STANDARD_HOME_OWNER",
        "2aeb18bb-fa1a-48d1-837a-ad084de2345e":"STANDARD_RELATIONSHIP",
        "1df82939-0545-40e3-877c-42207b8be7f7":"STANDARD_HHI",
        "ce351775-e609-48c3-81a6-6c0ea4136a4b":"STANDARD_COMPANY_REVENUE",
        "613da118-655a-4cb5-9d28-587e8900409b":"STANDARD_COMPANY_DEPARTMENT",
        "e1554555-bc7d-463b-b74b-9920a8ae4c0c":"STANDARD_NO_OF_EMPLOYEES",
        "b016327a-578c-4d75-8324-d777d089acb2":"STANDARD_EDUCATION",
        "edcd3fc1-201d-4b43-8f81-0585dcf16e6d":"ZIP",
        "c26f0d90-a0ce-4493-92d6-913900631686":"ETHNICITY",
        "7222d4e7-cae3-480b-b69e-f6b9d91e706f":"STANDARD_CAR_OWNER",
        "4cdf1000-79bf-4ea0-ae26-469aac01812e":"STATE",
        "3404dd6e-df23-4050-874c-55dd468bc27e":"REGION",
        "3eaf5ea3-bf00-48b3-a2a4-b07fefae03a6":"PROVINCE_OR_TERRITORY_OF_CANADA",
        "bf3d2a7c-2014-4aa9-a0a1-48422baf7a18":"AGE",
        "86e12c70-5c49-4c67-b238-6a5b189537f5":"STANDARD_INDUSTRY_PERSONAL",
        "385c2af8-d735-408f-889a-5a5cb4f7a2c6":"STANDARD_JOB_TITLE",
        "475f2d60-0f53-4ffb-8b80-628ab1491465":"STANDARD_B2B_DECISION_MAKER",
        "d49ea032-99ea-4a17-89f9-802ca8653bca":"DMA",
    }

    d = {
        "c0204685-206d-40e9-9c7f-22b3ca03de66":"Boy under age 1",
        "28b06a27-023e-465e-a262-c2bc8a93d07c":"Girl under age 1",
        "bf8ed8c4-0599-418c-88bf-0fece23e0a38":"Boy age 1",
        "e4ada3ad-3797-4286-b573-73f22e37fd14":"Girl age 1",
        "0c15c12f-b078-4610-92b2-0555cffb0486":"Boy age 2",
        "96bb888c-467e-44bc-8c84-36eb07b0004f":"Girl age 2",
        "9a73e388-24d2-4434-afba-256c0646e882":"Boy age 3",
        "17f7aaef-2360-405d-bc5b-87efe5f833a1":"Girl age 3",
        "605faad4-8035-4bd4-be67-2e90862a82fe":"Boy age 4",
        "049ffa4e-e8fa-4790-8c7f-5a6e815afade":"Girl age 4",
        "a8c44733-f3ab-4e40-9fa0-104a2c8ca675":"Boy age 5",
        "28ed6601-4b32-4020-be20-e59174e632b9":"Girl age 5",
        "60a321f0-fd13-4d4c-9f64-c549fd50b549":"Boy age 6",
        "66ac4c3b-b6a5-40b7-8f5b-af8e6586af78":"Girl age 6",
        "791d9214-ba0f-469e-bcc3-28865b5c0c39":"Boy age 7",
        "92be8626-f471-43e2-af12-54e2846cdbc3":"Girl age 7",
        "66678831-f2b9-4fc2-a3e6-b940fab73b82":"Boy age 8",
        "46840adc-885c-4cb5-b7f7-0509184bf26c":"Girl age 8",
        "5648fead-96f2-4930-9b8e-859bfcdb5f95":"Boy age 9",
        "09cd27d9-627b-424e-9ec4-0bd5636d3d2a":"Girl age 9",
        "d0de17ea-a74e-4ca4-aa86-d4dc9c03ceb1":"Boy age 10",
        "c260a77e-4af4-4e27-a92c-e90b09798bc3":"Girl age 10",
        "0345f7de-43b2-4000-8984-5579d39556b8":"Boy age 11",
        "e92960f3-32a5-4471-a546-f8a384d1b4b0":"Girl age 11",
        "87ac8d7b-cab5-400c-9b41-9ec4031ff898":"Boy age 12",
        "7b036941-790e-46fa-941f-b04499cb3877":"Girl age 12",
        "248b8bb5-b19a-4430-b72f-eb1f3bd51df6":"Male teen age 13",
        "e000581d-37f0-446f-8a3a-fb959179bddb":"Female teen age 13",
        "94b6dae7-9585-4d02-a415-07983650ab77":"Male teen age 14",
        "bcc29115-330a-41ce-bd97-b1a341e6c03b":"PORTLAND-AUBURN | 500",
        "f4ce8f25-c586-4fe7-9df1-96dc82e6122f":"NEW YORK | 501",
        "968e3c59-3e5d-4db1-a1f5-d7f57a5b2497":"BINGHAMTON | 502",
        "c748f4e5-e6c3-40aa-90a6-1128e4c760b5":"MACON | 503",
        "b4855278-3e0c-4b0c-93c3-f23cd32f278c":"PHILADELPHIA | 504",
        "6e1ac532-f4e2-49db-b935-91b7feec5657":"BOSTON-MANCHESTER | 506",
        "6d1b1265-c8f4-4f3f-b1ca-358a351c394d":"SAVANNAH | 507",
        "e87f20eb-6adb-4228-822a-5c68179af428":"PITTSBURGH | 508",
        "ebdc7e78-6f97-4415-96f9-d45d8a638c93":"FT. WAYNE | 509",
        "b4f63c0a-6376-4413-80bc-1820d9e21ab7":"CLEVELAND-AKRON-CANTON | 510",
        "4a3af34a-aa0e-4488-837a-69bc1503da4e":"WASHINGTON, DC-HAGARSTOWN | 511",
        "a0f9168d-0626-4220-bcd9-e6879b194780":"BALTIMORE | 512",
        "c72b8ffd-964d-4276-85a4-ea1bdd335606":"FLINT-SAGINAW-BAY CITY | 513",
        "3be9c6a2-009e-4e01-8886-72c63ba16b0a":"BUFFALO | 514",
        "77c32b39-7e33-4974-ab6b-8d851c0555ab":"CINCINNATI | 515",
        "381d3a23-b481-46cd-9cc5-882dfbd16944":"ERIE | 516",
        "69ab8aa4-352b-46d5-bd1b-a7a4b100e905":"CHARLOTTE | 517",
        "81d4abed-c034-444f-a9d3-beadd69a80b0":"GREENSBORO-HIGH.POINT-WINSTON.SALEM | 518",
        "d2e05982-ea91-4cc3-afcd-d73d3798fffb":"CHARLESTON, SC | 519",
        "f0ac3318-ab49-4b05-a522-cbbfdf0fd2f7":"AUGUSTA | 520",
        "f515c3c4-cce8-4c7e-aef5-cdda6c541ff7":"PROVIDENCE-NEW BEDFORD | 521",
        "f3ec509c-3dc7-4884-b5a6-de87705519be":"COLUMBUS, GA-OPELIKA, AL | 522",
        "0ee999e1-242a-4d68-8423-14e799ec148c":"BURLINGTON-PLATTSBURGH | 523",
        "f227d518-439f-41af-8d30-e01a296159c0":"ATLANTA | 524",
        "5da2946b-34d4-4f1f-9539-b91f7184578a":"ALBANY, GA | 525",
        "355a3c27-ad8f-4649-9114-6cf477baf1fb":"UTICA | 526",
        "b6d5f59c-466a-4766-bea5-a7b272d525ed":"INDIANAPOLIS | 527",
        "cd844db3-bbcb-4463-a423-7b14fa4a5d21":"MIAMI-FT. LAUDERDALE | 528",
        "b7c927b7-302f-483c-88a2-f9b1459b6793":"LOUISVILLE | 529",
        "15e54f5b-d611-4f0c-b8ba-fa584470d7cf":"No , not of Hispanic, Latino, or Spanish origin",
        "b13997c6-7be0-484e-bef6-b8b032e9083e":"Yes, Mexican, Mexican American, Chicano",
        "2de826c7-5d6a-4466-81bc-003be00d7f22":"Yes, Cuban",
        "7a5a3b67-fab8-4ce0-8dd7-0925c74c250c":"Yes, another Hispanic, Latino, or Spanish origin *** Argentina",
        "85bca147-e1c1-4500-9e3a-c2daa1edeba5":"Yes, another Hispanic, Latino, or Spanish origin *** Colombia",
        "8f35ae8c-6990-4b7b-ad4e-08872260b46f":"Yes, another Hispanic, Latino, or Spanish origin *** Ecuador",
        "fa552766-2ad0-42f7-903b-bc041718dec0":"Yes, another Hispanic, Latino, or Spanish origin *** El Salvadore",
        "8final_df1a3e-b4f9-4ec0-8c24-33c819fad067":"Yes, another Hispanic, Latino, or Spanish origin *** Guatemala",
        "7a663b59-8394-4a04-830f-c413faf7b6be":"Yes, another Hispanic, Latino, or Spanish origin *** Nicaragua",
        "3a807f0c-7899-4d53-a91b-61a1a64fa468":"Yes, another Hispanic, Latino, or Spanish origin *** Panama",
        "5b217ccd-1fe6-415d-b598-358927e0a663":"Yes, another Hispanic, Latino, or Spanish origin *** Peru",
        "dd2c3283-c00b-45a0-8e84-78bf41622a02":"Yes, another Hispanic, Latino, or Spanish origin *** Spain",
        "ba5241f6-9f4d-49f3-b151-bf096bd73fe1":"Yes, another Hispanic, Latino, or Spanish origin *** Venezuela",
        "8f583370-b062-48e0-b74b-7408d40cd966":"Yes, another Hispanic, Latino, or Spanish origin *** Other Country",
        "f26fe853-96aa-4636-a754-ed6b213f625e":"Yes, Puerto Rican",
        "82db9bcc-ac31-4419-83ee-bb2b5a42418f":"Prefer not to answer",
        "327d33c1-a936-411e-9f04-4c4249ec9315":"COPD",
        "58055408-ba37-48f4-9273-8902ed48814c":"Cirrhosis",
        "8d4dcd76-e7a9-4f70-b224-bd8c918620a1":"Rented apartment",
        "ed2c20d2-f70c-4f2b-9485-78e4ea748011":"Owned apartment",
        "5c903d99-4bf2-402c-82fd-9127799e9fe6":"Rented house",
        "bbf94d70-f354-4e6e-9d18-eddd16c44a7a":"Owned house",
        "d7ee5692-d808-481e-a5b2-439fc32352a8":"Farm",
        "a911aa61-fcc4-4f45-8eb9-4abd4e9b2b5f":"University Residence",
        "1a174845-0932-4c96-90e8-0e8e21dfee77":"Living with parents",
        "fd6f34cd-b7ce-432d-a0c4-2e0dd0195bc0":"Other",
        "6fa31727-c7b2-4411-b973-aed5518daeb3":"Single, never married",
        "395935a7-4df4-4f60-aa06-b6d97ee8bff0":"Married",
        "e56e91fe-a88e-4684-be74-79e19616ba6f":"Separated, divorced or widowed",
        "1fde83c1-a836-484b-8f8b-3e0fc7137dd0":"Domestic partnership\/living with someone",
        "674059b5-072a-4455-a55f-e7d613120e6a":"Prefer not to answer",
        "b2bf734d-fd45-4500-b6e6-4e23d1c8481c":"Under $100,000",
        "baebd5b4-b952-41d1-bf21-dc6d22908a3d":"$100,000 - $249,999",
        "90cbd532-5fd9-423e-bb6a-89c690850e99":"$250,000 - $499,999",
        "de92b989-9fa1-42ce-a200-8c2b8a1b73b4":"$500,000 - $999,999",
        "2b7be8fa-c52c-4dc6-9ec2-abcd10e38680":"$1 Million - $4.99 Million",
        "3d4d7791-0144-4d78-b7e3-27b29317d9c1":"$5 Million - $9.99 Million",
        "eb36e5de-c98a-4670-a0a6-7c529dfe73fa":"$10 Million - $24.99 Million",
        "6f992c00-67b3-4be0-9ffb-f4855891711b":"$25 Million - $49.99  Million",
        "e0f10553-048f-4053-9612-deffeb5192eb":"$50 Million - $99.99 Million",
        "eebbcf9c-9cdd-4842-8a61-c329154b5059":"$100 Million - $249.99 Million",
        "647f9ff1-6bc2-4451-aaad-32500903fb15":"$250 Million - $499.99 Million",
        "8079623b-598c-42fe-a957-89a562b5c65b":"$500 Million - $999.99 Million",
        "48cf4381-eecc-41d1-9643-b70f28886bd3":"$1 Billion or more",
        "dfe63a76-575f-44d1-918c-ec475475b5ca":"I don't work \/ I don't know",
        "1f847f14-c719-4687-b265-d4ed1ef28933":"White",
        "58cd312c-f874-4449-886d-73436e20b9cc":"Black, or African American",
        "a95f89b8-f2b5-4dd5-a66d-680089777872":"American Indian or Alaska Native",
        "61222503-c0ae-4874-a121-e8dc986833e8":"Own",
        "9605d813-3c80-41b1-88f9-25133bc03681":"Asian *** Asian Indian",
        "a1a014e2-0957-479c-b45f-9602d6ccdb99":"Asian *** Chinese",
        "8b0fb76d-1a8d-4959-ad19-9c5fd3eb6b48":"Asian *** Filipino",
        "d6ba6b8f-d0ca-4a35-a0be-0cb601final_dfa02":"Asian *** Japanese",
        "db05c41d-bf76-4409-a353-d0933e42f9be":"Asian *** Korean",
        "7a5f7b78-2d8d-45d1-850e-4b312662c246":"Asian *** Vietnamese",
        "bd9371fa-417a-415c-94a2-03c8dc68227e":"Asian *** Other",
        "af18a68e-99ad-48a1-a098-094a9c97170a":"Pacific Islander *** Native Hawaiian",
        "72c36e7b-a294-484f-92a4-2d17fb2422b6":"Pacific Islander *** Guamanian",
        "2c616008-1ac1-4fa6-ac99-0f1abaaac744":"Pacific Islander *** Samoan",
        "6a2a5bc4-cd6e-4575-9368-e18f73df1a79":"Pacific Islander *** Other Pacific Islander",
        "dcd1d8ac-b662-40d2-80a0-f0a33bb7d732":"Some other race",
        "3f232f71-d6bf-45dc-91c2-910f0a6a6dc3":"Prefer not to answer",
        "88e8b73b-11f0-4bf3-ac65-aa478ce95b5b":"Yes, I own a car\/cars",
        "288118f6-329e-41d6-bce4-f2508e937e46":"Yes, I have access to a car\/cars",
        "70550749-90f9-4d4b-9d1c-a99263914785":"Yes, I lease\/have a company car",
        "af090829-d869-43e4-b6a6-b65632a4d6f5":"No, I don't have access to a car",
        "b4fbada0-63a6-4770-842b-f80722dffe7e":"Prefer not to say",
        "663cf3c5-1d4b-41e7-8956-4e2ce740600c":"TALLAHASSEE-THOMASVILLE | 530",
        "02535de6-03a5-4e50-817b-805a0a7b83bf":"TRI-CITIES, TN-VA | 531",
        "07fdbc82-9597-4786-b093-d2aa0c413a6f":"ALBANY-SCHENECTADY-TROY | 532",
        "9ff5c47c-6c72-422f-8956-226bd873511d":"HARTFORD-NEW HAVEN | 533",
        "6f9e8b14-4869-4abc-94ad-6c4e5680ee14":"ORLANDO-DAYTONA BEACH-MELBOURNE | 534",
        "c343a53b-b2eb-40de-84aa-4b32e08c77c1":"COLUMBUS, OH | 535",
        "ca4e006a-3e16-4b5d-b613-5e76a6d2daf7":"BANGOR | 537",
        "b2c96e31-cc89-4ae7-ac9c-cea71c962bf1":"ROCHESTER, NY | 538",
        "81b57b2a-927f-4786-8b71-6cb2473ab4c3":"TAMPA-ST. PETERSBURG-SARASOTA | 539",
        "c997df17-15ea-48f9-a88c-7b2b39a1aa1a":"TRAVERSE CITY-CADILLAC | 540",
        "3fe3c06e-012a-4da0-881d-2dbb6747d4a1":"LEXINGTON | 541",
        "07df8a82-b532-40a7-8a22-e00f33707a61":"DAYTON | 542",
        "2b9042f9-9222-48f1-a16f-0b24d29161bb":"SPRINGFIELD-HOLYOKE | 543",
        "ec44376b-8c91-49a5-a4c2-d8de779fee0c":"NORFOLK-PORTSMITH-NEWPORT NEWS | 544",
        "b50762c0-f8c7-4774-898c-79b7cc71cbbe":"GREENVILLE-NEW BERN-WASHINGTON,NC | 545",
        "70b95bca-6f82-4a25-b777-7154a10b34ae":"COLUMBIA, SC | 546",
        "d5150c2a-ca57-4ee0-b854-c8627e682398":"TOLEDO | 547",
        "bb908063-fdd9-45c5-a29c-e51f540c82a4":"WEST PALM BEACH-FT. PIERCE | 548",
        "72a97fef-6e4d-440b-8c86-b37dc5e09c0d":"WATERTOWN | 549",
        "6e12cb09-c605-475a-8e74-847b4f7ba9cd":"LANSING | 551",
        "d226a1cf-1ec9-4f49-96a2-aee0fa05fbe6":"PRESQUE ISLE | 552",
        "d41d1659-c562-45f1-a4ea-4db4be0f70ab":"MARQUETTE | 553",
        "d02a68c4-a2d8-4ff3-93ea-9a62bb6340f4":"SYRACUSE | 555",
        "a5663874-3651-480d-a710-721e0114a2cb":"RICHMOND-PETERSBURG | 556",
        "55faf6cd-2156-42de-ac1a-93f35464386e":"KNOXVILLE | 557",
        "c44872fa-332e-4599-9adc-cf9e8a55ae06":"LIMA | 558",
        "997d5023-2d70-43b7-8bd0-7133b23ec9ec":"BLUEFIELD-BECKLEY-OAK HILL | 559",
        "6411eca6-e56d-4e0e-a96b-a24e858e088e":"RALEIGH-DURHAM-FAYETTEVLLE | 560",
        "6509b677-f65e-4398-98c4-be95aec11ade":"JACKSONVILLE | 561",
        "5aae3139-3327-4b8a-adb4-9ca58bad97ad":"GRAND RAPIDS-KALMZOO-BATTLECREEK | 563",
        "9134e4d1-81e4-4af0-ac89-a3c46f41343e":"CHARLESTON-HUNTINGTON | 564",
        "0577f4ca-eff3-4b34-b45e-e130f9375423":"ELMIRA-CORNING | 565",
        "ace86f04-51a4-413f-b022-304f69994228":"HARRISBURG-LANCASTER-LEBANON-YORK,PA | 566",
        "09c6509d-02c6-4238-9ce7-86a7f68963b3":"GREENVILLE-SPARTA-ASHEVILLE | 567",
        "57f200cd-7c6c-4ccf-8a6a-28e0ac41cef7":"HARRISONBURG | 569",
        "d553c4e1-64c3-495a-8461-cce3de5e96f4":"MYRTLE BEACH-FLORENCE | 570",
        "87ef3759-5dee-4595-a069-fec736379131":"FT. MYERS-NAPLES | 571",
        "e1d91fa4-356c-4be0-b3e6-e1c99a0e826b":"ROANOKE-LYNCHBURG | 573",
        "a1f85242-a7d2-4a60-b145-4763cca93007":"JOHNSTOWN-ALTOONA-ST COLGE | 574",
        "7bca5e2a-a644-4c5d-8e34-4d706e4d334a":"CHATTANOOGA | 575",
        "2199bb19-ac68-4918-ac8a-721f75c7faa0":"SALISBURY | 576",
        "fee04be3-e319-49ab-80e5-380ac2aacdbf":"TERRE HAUTE | 581",
        "4cc00ddf-d01c-46b8-a386-d707a59c7fff":"LAFAYETTE, IN | 582",
        "b148e0bf-6b05-485c-8d95-a681b1f8033b":"ALPENA | 583",
        "9dfc8e6e-e66d-429f-9f7a-40e24320b862":"CHARLOTTESVILLE | 584",
        "2c20a9ea-5f24-4c9c-886e-9396025c677f":"SOUTH BEND-ELKHART | 588",
        "760c549b-d919-48b8-8268-24c20d778b34":"GAINESVILLE | 592",
        "5e17f553-a22d-4063-bbd1-012cb35a6bd3":"PARKERSBURG | 597",
        "e756d64b-4bdc-4674-a847-b692f668673d":"CLARKSBURG-WESTON | 598",
        "ee236ced-13e3-444a-818c-f32d7b23916f":"CORPUS CHRISTI | 600",
        "c0d67bfd-468c-4e94-9a0e-b61d5896fca0":"CHICAGO | 602",
        "13d44287-dca0-4d46-b248-c37bd0aeaa25":"JOPLIN-PITTSBURG | 603",
        "2d33caac-a5e4-49ae-b878-5e4cf50ebb81":"COLUMBIA-JEFFERSON CITY | 604",
        "9928ba26-3fb7-40d7-b68d-c24e6c3393cd":"TOPEKA | 605",
        "f4c4fdbf-0d37-4363-8963-fd9614876937":"DOTHAN | 606",
        "5820dd99-c4f4-4d70-9670-7c73f6668b28":"ST. LOUIS | 609",
        "a79bd2d1-c3f4-480f-aecf-48c641582018":"ROCKFORD | 610",
        "64dab6eb-cd44-4ef5-bf36-a75713cf12c3":"ROCHESTR-MASON CITY-AUSTIN | 611",
        "3c58d1c4-3f7a-431e-b549-377a8cbe5301":"SHREVEPORT | 612",
        "f7651feb-cf92-4f01-b65e-6dc313c26fd7":"MINNEAPOLIS-ST. PAUL | 613",
        "97187862-f06f-46f0-90d6-c953cb1b24db":"KANSAS CITY | 616",
        "febccbf5-b8f7-4928-b069-4ec234736d93":"MILWAUKEE | 617",
        "acbe2008-991e-4672-970a-d052db612213":"HOUSTON | 618",
        "96636ef6-e288-490c-9cac-77edaa854ee1":"SPRINGFIELD, MO | 619",
        "c5af92eb-fd12-4b91-9c12-4e41d5ce694f":"NEW ORLEANS | 622",
        "340de753-6855-45dd-9680-f4cdf0957c25":"DALLAS-FT. WORTH | 623",
        "dfe18403-5ed4-45f5-ab4f-143204d71990":"SIOUX CITY | 624",
        "10d8ff27-cef7-4eee-8ba0-df6f8ef0bded":"WACO-TEMPLE-BRYAN | 625",
        "ffafaa2d-7243-4d91-9b2c-623ce3610306":"VICTORIA | 626",
        "73712fee-9cd3-4137-8e25-e5e73adcc66a":"MONROE-EL DORADO | 628",
        "3faa6778-1eee-409f-96e0-93553b9d74e9":"BIRMINGHAM-ANNISTON-TUSCALOOSA | 630",
        "54f372bb-04c8-4349-98e2-304792f49408":"OTTUMWA-KIRKSVILLE | 631",
        "1c3bcac3-b434-4242-b20b-991e23bef56d":"PADUCAH-CAPE GIRARDO-HARRISBURG | 632",
        "d1fbd2a3-1c7a-4a11-9f9e-46a3d2e7a8cc":"ODESSA-MIDLAND | 633",
        "2c0b28c3-6ef7-4b1e-8baf-a449c973e1ce":"AMARILLO | 634",
        "4aaf9c69-48a8-4ba6-91dd-e0aead421f34":"AUSTIN | 635",
        "47262065-7f70-4875-929c-0a54dddb1f71":"HARLINGEN-WESLACO-BROWNSVILLE-MCARTHUR | 636",
        "fc1d2e3f-da6d-4a4e-bc65-879dbfbbd4d6":"CEDAR RAPIDS-WATERLOO-IOWA CITY & DUBUQUE, IA | 637",
        "4ac6b177-cb2c-4db8-8887-dcb040791434":"ST. JOSEPH | 638",
        "c019c279-36d0-42fa-9ac9-f83ac3be60d1":"JACKSON, TN | 639",
        "e8aff6cc-e207-4090-809b-d69d37bccfe1":"MEMPHIS | 640",
        "a2470bdd-e6e4-4d36-a232-f99d27f026f6":"SAN ANTONIO | 641",
        "8c5dd168-baf4-481a-a096-db6b68abb1e5":"LAFAYETTE, LA | 642",
        "071760dd-e89f-4412-82c3-fdd5e2d589f8":"LAKE CHARLES | 643",
        "6e259175-f934-4482-8c97-cf4463a73dfd":"ALEXANDRIA, LA | 644",
        "e89ec384-5ea1-41f8-81c3-9e663f92904b":"GREENWOOD-GREENVILLE | 647",
        "418368de-f3ed-45a9-9e21-7556044595e6":"CHAMPAIGN & SPRINGFIELD-DECATUR | 648",
        "bfe4ad65-a434-4fc0-94f4-832dbf78d5e5":"EVANSVILLE | 649",
        "a3eda141-ee16-43e9-adcc-860243b2fea6":"OKLAHOMA CITY | 650",
        "42a1672c-090f-41cd-a381-5454c60f3981":"LUBBOCK | 651",
        "6ee0bf47-4b05-4726-81f8-03b0f769289e":"OMAHA | 652",
        "c244595c-9f53-43f5-b980-10109071778a":"PANAMA CITY | 656",
        "715ed61e-8090-4b6c-ad67-89eabcdb39f2":"SHERMAN-ADA | 657",
        "97f3f8ed-6921-4b43-96ad-230887c38966":"GREEN BAY-APPLETON | 658",
        "4f78a461-dada-45bf-beb2-9f18d8abe648":"NASHVILLE | 659",
        "2a4bc28c-6eda-41db-8405-b208a94e7bed":"SAN ANGELO | 661",
        "0ed8f06f-7a32-4971-a6da-2c13170020e6":"ABILENE-SWEETWATER | 662",
        "24abf4d4-feaf-486f-9b94-670283808d80":"MADISON | 669",
        "6c3830c6-afb7-41a2-8592-1f2c55a651ea":"FT. SMITH-FAYETTEVILLE-SPRNGDALE-ROGERS | 670",
        "eae88435-a69a-4a5a-989b-f0f86fec461a":"TULSA | 671",
        "bbfadc01-588a-4da7-8ed3-afaff356db0c":"COLUMBUS-TUPELO-WEST POINT | 673",
        "b2301a81-dad5-42af-b254-4ddee3cf19b9":"PEORIA-BLOOMINGTON | 675",
        "1ce0fb22-fe3b-4de3-8991-82c5b90a9d03":"DULUTH-SUPERIOR | 676",
        "613f4049-50ef-4f47-a4fe-b5aec1a84760":"DES MOINES-AMES | 679",
        "5c78e1ee-28bc-4a72-ac10-f193f6a49f4d":"DAVENPORT-R.ISLAND-MOLINE | 682",
        "4e24c5f5-2945-4e0b-8602-cdae9427484d":"MOBILE-PENSACOLA | 686",
        "963c102e-4ed4-4ae5-b735-a59c75f4364b":"MINOT-BISMARCK-DICKINSON | 687",
        "81070e38-ea1e-4009-a230-cd1635035752":"HUNTSVILLE-DECATUR-FLORENCE | 691",
        "be042d67-b3c4-442b-bbdb-56fa9fb5785c":"BEAUMONT-PORT ARTHUR | 692",
        "11325d42-49c3-4e91-bae1-2c7714a6f24f":"LITTLE ROCK-PINE BLUFF | 693",
        "ef77e9d6-67b0-4548-a268-be8dae528d52":"MONTGOMERY-SELMA | 698",
        "1c2b321c-00c5-4bc9-a317-792149a3949e":"LA CROSSE-EAU CLAIRE | 702",
        "30b1751f-9699-407b-befa-5f557be351a6":"WAUSAU-RHINELANDER | 705",
        "780be38d-8184-4078-875e-b49467f51e40":"TYLER-LONGVIEW (LUFKIN&NACOGDOCHES) | 709",
        "5b82c8b6-3d7e-44d5-a355-9393110a7e93":"HATTIESBURG-LAUREL | 710",
        "5b8c274d-932b-4d17-9019-8d1a333a296a":"MERIDIAN | 711",
        "f258d16b-8a38-44f9-875e-2ca164a668a5":"BATON ROUGE | 716",
        "8ebd1ac7-a357-4b20-bc8a-91ef63c9405f":"QUINCY-HANNIBAL-KEOKUK | 717",
        "980f820d-23f8-4369-8a2c-09955075a56f":"JACKSON, MS | 718",
        "73fa9aae-b165-4c54-8813-f601f52ca1dd":"LINCOLN & HASTINGS-KEARNY | 722",
        "7c7bae35-762a-4b76-8a84-b200c32ac2af":"FARGO-VALLEY CITY | 724",
        "a3b3b361-fb96-46a6-bcfc-3d86e1fef76e":"SIOUX FALLS-MITCHELL | 725",
        "129c7b4e-513a-426a-8c5e-a03696d111be":"JONESBORO | 734",
        "4682a249-843d-409b-812d-ee980a53089d":"BOWLING GREEN | 736",
        "23183439-67b3-4e38-8f62-3ff9d3c3fafa":"MANKATO | 737",
        "863e32e1-2931-4f64-b491-68da1261669e":"NORTH PLATTE | 740",
        "3b44dcaa-6289-40d1-b077-8ce5f43c9966":"ANCHORAGE | 743",
        "323ad20b-f88a-49d4-8b70-f7f992524d55":"HONOLULU | 744",
        "d53efdfd-fbe8-463a-bb36-979c45b967da":"FAIRBANKS | 745",
        "d294c747-6011-4a05-a1be-fd2991bab93e":"BILOXI-GULFPORT | 746",
        "7659ee67-7c3d-4e6e-a6d0-893f37202e09":"JUNEAU | 747",
        "a2506332-831e-45bb-95e3-de801c0a485c":"LAREDO | 749",
        "01321c41-a83a-4641-b48f-60bc4a9c2090":"DENVER | 751",
        "f1b3f44e-69ff-43a4-857f-4193be106495":"COLORADO SPRINGS-PUEBLO | 752",
        "f5ba6aa3-ce14-42fb-8de5-08bd2d586eb0":"PHOENIX-PRESCOTT | 753",
        "b6eb1966-f6f8-42cb-b16a-7ae9a3b1bea4":"BUTTE-BOZEMAN | 754",
        "8138e952-aac5-4b17-9aa1-dab77ca2a758":"GREAT FALLS | 755",
        "1c2ad258-fc8e-43f9-a9e9-2da31d23ce60":"BILLINGS | 756",
        "209f93e9-6bae-4aee-92ca-7f8a08bd8f74":"BOISE | 757",
        "d6d14c5d-4980-4d00-aec1-7fb2090b601a":"IDAHO FALLS-POCATELLO-JACKSON | 758",
        "0ac0c144-db9e-4186-b379-cef7e594ded1":"CHEYENNE-SCOTTSBLUFF | 759",
        "c358846a-17cd-49f2-9f79-d6bd3f3d6416":"TWIN FALLS | 760",
        "8287e938-04e1-4fba-9e09-e86341d4a7d1":"MISSOULA | 762",
        "403607cd-65a5-467c-9fae-a5d807d425df":"RAPID CITY | 764",
        "b9c5d882-a1bc-4a23-b006-e8ff2e841d1f":"EL PASO-LAS CRUCES | 765",
        "521b9da0-2af9-4209-bcfa-f5bac5b39b26":"HELENA | 766",
        "5a708082-45e9-4657-a5c1-ad55b7150ad3":"CASPER-RIVERTON | 767",
        "9045f1de-6c2a-4404-83a2-31893710fc52":"SALT LAKE CITY | 770",
        "e76bc11b-575f-4423-8f0a-65dbd92dc14b":"GRAND JUNCTION-MONTROSE | 773",
        "faee3e68-9d3f-4c3a-9c7e-8e6a8ecd7126":"TUCSON-SIERRA VISTA | 789",
        "d20a7f5a-84e9-42ce-8ec0-a1e3b56cc1e7":"ALBUQUERQUE-SANTA FE | 790",
        "7680ddfd-2d03-4bc7-a501-ebaa739176bb":"GLENDIVE | 798",
        "52497810-c658-4fc1-97f1-964a47d2db6a":"BAKERSFIELD | 800",
        "8b5bcce1-fdfa-4b2e-a09c-fb27adb819b7":"EUGENE | 801",
        "042e33b4-5336-4fcc-9e39-2c5742051edd":"EUREKA | 802",
        "40af6e05-a77e-45fc-9626-d0569ed5980d":"LOS ANGELES | 803",
        "98ecca7e-742f-4add-86f8-34018a5ef741":"PALM SPRINGS | 804",
        "bbe391c8-2f7e-4448-b345-03e3305c7a7b":"SAN FRANCISCO-OAKLAND-SAN JOSE | 807",
        "fc0e6ee7-d939-4876-ad94-d7633543fdc0":"RENO | 811",
        "16ba9e98-4e42-4cf5-b852-94e0886d04f0":"MEDFORD-KLAMATH FALLS | 813",
        "42e1806a-95ba-47f4-91f6-047175f3f7d3":"SEATTLE-TACOMA | 819",
        "ba885a09-9bdb-478e-82dc-b799a161b4b4":"PORTLAND, OR | 820",
        "7aa1b773-55f9-44f0-8b07-1468c697f5da":"BEND, OR | 821",
        "722ee321-8415-4b38-ba0b-c5bb00ec1aa4":"SAN DIEGO | 825",
        "fb0730e0-064c-4c31-9aca-3971918e801d":"MONTEREY-SALINAS | 828",
        "de919a63-6ea7-440b-97c1-b73816895a03":"LAS VEGAS | 839",
        "34e83bb2-eea4-4d5e-8547-6d52b4bdd1f4":"SANTA BARBARA-SANTA MARIA-SAN LUIS OBISPO | 855",
        "0ce008b9-9a33-4c4e-894e-9806d4f42da2":"SACRAMENTO-STOCKTON-MODESTO | 862",
        "67efc495-bdd3-46d8-a1cd-2bbe72616a94":"FRESNO-VISALIA | 866",
        "ef64d8e3-2630-4f49-8d76-52cc5580381e":"CHICO-REDDING | 868",
        "6a10a0c1-1146-44bd-a754-d06c6ebb45fc":"SPOKANE | 881",
        "3b3f60ce-fb04-4cca-8aab-88192d9535f9":"Colitis",
        "e9417bf2-1521-4b3a-91f4-06b1b501e8a3":"Constipation",
        "1bfae6b1-7dc5-4ba2-9d01-decc55141882":"Crohn's Disease",
        "d2a7c695-a9b1-4c55-8751-f9d26800fb24":"Cystic Fibrosis",
        "aeba59fb-ac17-46a5-9382-dbfa20f0e72e":"Dementia",
        "5e431d2a-5656-417b-9ae8-a5e622ceea25":"Dental Problems",
        "d9d4d577-15c7-4b85-85c3-62590fb08618":"Depression",
        "079384cb-31c1-4faa-aaec-77c0043718f5":"Diabetes",
        "062e0243-48fb-4750-ab4c-2508c106e267":"Diarrhea\/Diarrhoea",
        "dc66391d-e3c1-4e99-822f-635a4ee156da":"Eczema",
        "8f7b4b94-7364-4f1d-adca-ba3dd9627b6e":"Emphysema",
        "812499c2-302c-4a9a-bcf5-f35763a8c043":"Endometriosis",
        "de7e2280-5868-4ae1-8933-9b06189d783b":"Epilepsy",
        "e12223f0-b09d-49c2-b3be-f3df993b3b5a":"Erectile Dysfunction (ED)",
        "98ecefe6-76f5-48da-9a20-275d805c6b2d":"Fibromyalgia",
        "9193fe85-6c31-4c72-8a40-9dec7f136258":"Food Intolerances",
        "1c980388-5d72-4a4f-b7d0-dffa2fe536a6":"Gastro Esophageal Reflux (GERD)",
        "280e02df-5c6f-40ec-b072-1668cb205079":"Gastroenteritis",
        "096c85d1-e046-4a2c-a010-d0c1bd74b5b6":"Gout",
        "c7ca0024-b450-4b93-8459-92e845cb7f8b":"Hemophilia",
        "dba0e83c-72ac-4137-b09d-645d52a2558e":"Hemorrhoids",
        "6ea37a67-25f9-491d-8d82-3cb16e88e281":"Hay Fever",
        "a521fd8d-60c3-4ca1-af59-bf5b6608c677":"Heart Conditions (not heart failure)",
        "3112e7b1-f6e2-486d-b561-1c5bb854e880":"Heart Failure",
        "e7cf83fb-9fda-4d69-adfc-5c86b7504e1a":"I don't have any illnesses\/conditions",
        "aaf5527d-ae9d-4b4a-a952-12b13441c908":"Other",
        "e155ef55-af11-405d-a4e4-99f83b5504fa":"Prefer not to answer",
        "162f8356-37ff-4915-a00f-91858238087a":"None of the above",
        "b7941882-3780-4eb9-bd42-4860f44170dc":"KANSAS",
        "e090de3d-5c43-47e9-8e7c-0b7f2e03d11b":"MARYLAND",
        "76d78228-ea4f-402f-b102-7f3f06847f8d":"WEST VIRGINIA",
        "a1cf208d-6ca6-4db7-995c-ede55fdb4d56":"WISCONSIN",
        "ad826e15-e374-4b38-975b-85486e0d2de5":"WYOMING",
        "33a9530e-9402-4d2c-9e06-2551a5444fab":"Northern Mariana Islands",
        "8d70328a-443c-4c51-8685-28ae441a845c":"Palau",
        "753ba2e2-c3e4-4c3e-abe8-b4882d3ae82b":"Puerto Rico",
        "a01f8c5b-b4ef-4967-bc31-9ac0ff9a4983":"Virgin Islands",
        "32cdf98e-b983-4c48-8b14-0e629adb52d8":"NOT APPLICABLE",
        "66fea960-217c-44b2-a2ef-7e92fb104808":"Some high school or less",
        "3cd02fc9-0326-43ed-b669-a919f3edd20d":"Bachelor's degree",
        "d171fe93-a197-4161-8223-8d2d40814942":"Master's or professional degree",
        "00bc6e3f-0fe4-4145-afa1-fd883f6048b4":"None of the above",
        "ae7149ff-63db-4be9-b554-9412e3fbcb7b":"Associate's degree",
        "d3798ede-69f7-42f5-9676-44aa29a53b5c":"$50,000 to $54,999",
        "37068f8d-563a-48ea-9eee-ef4edc4510fe":"$55,000 to $59,999",
        "fe93a9b3-38a2-404d-941c-93457830f027":"$60,000 to $64,999",
        "88d32a46-3458-4f1e-a4ac-fd1b156a514b":"$65,000 to $69,999",
        "67b2b26f-f947-4027-b555-816e89744d93":"$70,000 to $74,999",
        "9f604f26-a1cc-42af-ac74-2a559d89c216":"$75,000 to $79,999",
        "8592cc55-15e4-4574-88e9-d459f8e632c5":"$80,000 to $84,999",
        "764846c1-79e8-4c33-bdf1-e774c6713319":"$85,000 to $89,999",
        "aafde646-ba5b-4121-8eb4-7b9bb3963ce5":"$90,000 to $94,999",
        "594f528a-ca26-4464-9b75-8ad90c6e1363":"$95,000 to $99,999",
        "502bdecf-c9a2-4af6-9538-7807a29d8e7d":"$100,000 to $124,999",
        "3eb38a20-3e6a-48fd-a45d-30920183f649":"$125,000 to $149,999",
        "2f485bcc-da49-4c68-bf51-5d0b32c6d1fe":"$150,000 to $174,999",
        "d9f65f81-ab1d-43bb-aaca-508ee7b798d4":"$175,000 to $199,999",
        "11f68263-24d5-4f16-bf59-b01ce4d8c374":"$200,000 to $249,999",
        "436707b8-c5ae-44d5-802b-88dbb745f75c":"$250,000 and above",
        "c6f56358-c7f2-468b-ad3b-a50f5f7d5ae0":"Prefer not to answer",
        "3e10d69b-6e28-4209-bbc3-56663405c371":"High school graduate",
        "43final_dfeb1-f830-4139-a2bf-c30bef934ad3":"Other post high school vocational training",
        "fe36db8a-b828-4a78-be7a-26725f167d82":"Completed some college, but no degree",
        "7ad9528d-7222-4411-9876-08ded128168a":"Doctorate degree",
        "9c837ecc-03c8-4e1b-b99e-f38a43a56c21":"Education",
        "cdc5a04e-a21a-4d6e-b9f8-dbca9bb0b037":"Energy\/Utilities\/Oil and Gas",
        "b518cf68-cd0b-4353-a201-2a87d442311b":"Engineering",
        "af59f8f3-ffd4-4d6d-ada1-2f764dd2614f":"Environmental Services",
        "b5a9390b-26eb-45b8-9328-fa0aa11c9445":"Fashion\/Apparel",
        "b5744a0e-a2fa-444d-a0d2-75cc01714d37":"Food\/Beverage",
        "c7f5216e-64e7-43be-b3d5-3afd3a81457c":"Government\/Public Sector",
        "efa645ba-73f3-4cbc-bf46-6feb6c79e72f":"Healthcare",
        "92805de0-1edb-4b6d-a7ea-9c98abbeb3a6":"Hospitality\/Tourism",
        "f90065bc-f99f-485b-bc78-7b1c0e558b5b":"Human Resources",
        "416f171b-25f3-44da-aafe-65db2b35cfca":"Information Technology\/IT",
        "7c4faa85-39e1-4fdb-9362-931d8ecd81b7":"Insurance",
        "6a23fd7d-7e03-4adc-b6b6-210275b6b037":"Internet",
        "526b80be-2139-4157-bc74-9da3b5387f0c":"Legal\/Law",
        "2749c589-6060-4899-9b56-c23c22f34dd2":"Manufacturing",
        "3dda30fc-1c19-4c32-bec4-038f44496af6":"Market Research",
        "741764b3-d579-482b-a018-e6942e1f7c94":"Marketing\/Sales",
        "c305d7f9-2be6-4159-90d7-4aad393dd2b1":"Media\/Entertainment",
        "03bd4fd2-6120-4b3c-8ba8-6906d7af9dca":"Military",
        "7cc309cf-e974-4f17-b8ba-bbbdeb5ea9ae":"Non Profit\/Social services",
        "d933d969-9728-4d0d-9afb-00382f1b3685":"Personal Services",
        "0203a3af-aef0-4fba-9b5a-3319ea824fc5":"Pharmaceuticals",
        "ec1b427c-f667-4535-9fa7-b4cfffe03057":"Printing Publishing",
        "63b878c1-a199-45e8-9527-aa2daf19164f":"Public Relations",
        "efd6abc7-6c9a-4d6d-92e3-fbaecec7770e":"Real Estate\/Property",
        "308053bb-7c86-4f0f-9396-c560ca525c3a":"Retail\/Wholesale trade",
        "54558ad1-d0e4-4394-811e-7443d7887288":"Security",
        "1aeabf4e-c4b5-43ce-9669-0733e5e3f190":"Shipping\/Distribution",
        "d19fb60d-ad1a-4306-976a-83ac5205b1d0":"Telecommunications",
        "a53131a8-1ae5-40d9-9594-7c1ecfa5f4ce":"Transportation",
        "742afeeb-5997-40f4-b6f2-fa1d497f3d51":"Other",
        "d92df0f7-42bd-4ad0-bb6a-8bc7be47d5b9":"I don't work",
        "f6ef07db-ac66-4834-8c1c-2dd437cefede":"Procurement",
        "88c5fbdb-b5f0-4310-b1f1-adfdd8c9b056":"Sales\/Business Development",
        "73a91b5b-bf64-4da8-ba94-b839381253bc":"Technology Development Hardware (not only IT)",
        "239bb0f2-03ed-4011-bd5c-87b60a66eff0":"Technology Development Software (not only IT)",
        "1e623d02-bea7-4078-8c71-fff640a6d140":"Technology Implementation",
        "9cad1661-3b99-47dd-a396-c492d9aa5d79":"Other",
        "34dd8deb-44dc-4023-8a2a-08e423f0bd67":"I don't work",
        "d6d1e749-1bfa-4444-8a1d-1ffb7f106381":"Assistant or Associate",
        "7f4ec555-f649-4d6e-9de8-d489bbf7e116":"Administrative (Clerical or Support Staff)",
        "62ae6355-6982-4f73-9b18-3040c13957c2":"Consultant",
        "b4ca8252-6b4b-48dd-bf91-c1d74744f5e6":"Volunteer",
        "d8104b74-a161-4d55-81b2-cad565c2296f":"None of the above",
        "c4403be8-d236-4020-ac1b-a958d179c6c9":"IT Hardware",
        "359bb208-2fdd-4e4f-88e4-2ff08f02f97e":"IT Software",
        "fc9dbd86-6e0d-4446-9b51-9282c9c56567":"Printers and copiers",
        "be9b51c5-6e4c-4210-a27f-88d23c5cade4":"Financial Department",
        "be601630-7e44-485b-9f90-268d54a8e37d":"Human Resources",
        "73612662-ca50-4745-a9f6-72c6f09b010a":"Office supplies",
        "b6df745b-0a31-4818-93f3-11c900426d30":"Corporate travel",
        "c526c6a2-f73a-410d-8a0d-fc1c6e98e6c1":"Telecommunications",
        "6bbf8552-03d1-45bb-8530-06dc5e7210d8":"Sales",
        "2fa10a55-2ab5-4421-923a-33bd2be2b911":"Shipping",
        "f5c7baf1-5fe3-4ece-bf65-5c1f05cd5b4f":"Operations",
        "6e685474-00a6-4408-ad35-daf63b10938d":"Legal services",
        "70595a2e-6dda-41c1-a614-ca293767e0a7":"Marketing\/Advertising",
        "c79851f1-6230-40e1-8124-f9d3b3bc98e7":"Security",
        "7b5c68f8-0467-4094-986b-4b253c2ea544":"Food services",
        "754eed6a-ee89-42f4-8b53-3b3e891d6443":"Auto leasing\/purchasing",
        "1a0836c2-8357-4f0f-a219-c96c31cf6c77":"Other",
        "311eb995-a2f9-42be-bc7b-427b8774450c":"I don't have influence or decision making authority",
        "266a8322-908c-4c0d-a9aa-3fc8bed29a41":"Acura",
        "1c0232e4-5f40-4fcc-bd20-4fa3af15ccd6":"Alfa Romeo",
        "71605af7-b6b5-43d3-be59-ca1e1136fffd":"Aston Martin",
        "3556b540-789c-403d-8d69-bf37aeba0931":"Audi",
        "ab1800a1-d680-42a4-80f5-c81560b1e764":"Bentley",
        "f1ad292d-1f98-4e38-aa93-8ca0412d6385":"BMW",
        "a9aec3a1-edff-4823-8e53-e5154129b7de":"Buick",
        "a8608843-0f10-453b-91bc-ae0eb0685ce7":"Cadillac",
        "38d718a0-8796-425a-b5cd-bd5a249b97fa":"Chevrolet",
        "75674acf-eb13-4476-8323-4b6b5abc7652":"Chrysler",
        "742e08ac-d9cd-4ca6-9f45-0ddc65edfe9e":"Dodge",
        "8f32f1af-6379-48f8-8dd0-c3fac2df684b":"Ferrari",
        "498e39d4-0a97-4e1f-818b-76188c1858c9":"Fiat",
        "b9fd6d26-b478-4741-acdf-a8747d37b52d":"Ford",
        "6ca03827-fd1b-44a0-8757-776c3e55f127":"Genesis",
        "43cac3e8-1417-4c2f-a8b4-4d2e3e807bf8":"GMC",
        "d0b85fdf-7556-4df8-bec0-3a14ac2080c3":"Honda",
        "30b8387d-e162-4fa0-ac1a-4ab079cf430f":"Hummer",
        "6a826ad3-32e9-404c-bd68-843e69b3b4b3":"Hyundai",
        "9014ad73-c004-456b-a3d4-58fba31e3c79":"Infiniti",
        "20aa8895-83f0-4774-9a52-5f18ce8c47c4":"Isuzu",
        "f0562247-1642-437f-9a52-f3a0f8bcb897":"Jaguar",
        "d73c4926-f678-40b1-aa2e-6b668c55fa17":"Jeep",
        "74d6bdf7-792b-4068-8257-cba659bb18e0":"Kia",
        "e1f4c52c-ac7c-4398-87f3-d9b8b6867681":"Lamborghini",
        "9302ec74-ac58-43ed-bc9d-8812eb4acfbb":"Land Rover",
        "cbe4006a-2884-4e1d-b0cf-afb8461e48b0":"Lexus",
        "56451f77-3fb9-4692-bd38-14f33fa42ed4":"Lincoln",
        "88c6fd47-8276-4e32-a5c7-ace68ac956c4":"Maserati",
        "9ee26ec0-d854-4370-8908-d1516ed1e90c":"Mazda",
        "bcf2e69c-a11e-42a5-90ea-8f37e6ab8db8":"McLaren",
        "13274d61-0951-40f2-83d6-92d3c28d4d14":"Mercedes",
        "8028327d-1b08-48ce-be53-432b65b4fd3e":"Mercury",
        "13c2ff00-23f8-4042-a4d2-2142da4ec58b":"MG",
        "6b26db51-2fc9-4ef9-9201-c2e50f3f989f":"Mini",
        "3d52b971-a884-4037-8b96-9413c95c9cce":"Mitsubishi",
        "39e64aa8-f1c5-46c6-ac7d-17def369d7af":"Nissan",
        "edc0290d-d44a-4465-b86b-3df9294534c4":"Pontiac",
        "4de80df8-8f04-47cc-b5d1-eeb99d62b4c2":"Porsche",
        "cee63105-6bc8-468f-914b-10872e5587ba":"Ram",
        "8eddfed3-ea3a-4350-81ae-75b83bed0fb3":"Rolls Royce",
        "128e2c9f-99c2-4a66-bc4d-de0b52c85cb7":"Rover",
        "36e1c615-e48c-461a-a92f-98b84e1c34c0":"Saab",
        "25526c8e-15d5-42c4-afc0-642e5135d598":"Saturn",
        "cc26f9f6-ddbf-4437-804c-e636a12eedcc":"Scion",
        "92fad5e4-d431-4b3a-a022-19b1b38724a0":"Smart",
        "0c5478d8-1980-49c3-b639-ce7e8b73bf50":"Subaru",
        "f0633c42-bc3b-475f-bf6a-e97a2ef83a59":"Suzuki",
        "932efa86-f112-4645-bfaa-37eedb8646e5":"Tesla",
        "16b7b45a-04cd-487d-8752-cb43b5145c63":"Toyota",
        "58ea9206-aaa4-4ee2-913b-54a71d18b56f":"Volkswagen",
        "54bc9b09-e1fb-428b-aa82-965aff37d2ec":"Volvo",
        "f071c5fa-c40b-400b-afe3-73d80c914cc9":"Other",
        "968c472a-a27a-4631-874e-2f369985f8cc":"I don't own\/lease a car",
        "e1f06623-4c52-49bd-a7b9-f8640e7779eb":"Employed full-time",
        "af879399-97ec-4cef-9b3e-4e199f5f024e":"Employed part-time",
        "21a98fff-04e5-4176-85d2-7fa73c4ac3f6":"Self-employed full-time",
        "1e6699c3-6b42-4048-85c4-7f1fef6129db":"Self-employed part-time",
        "9132eafb-222c-472d-8bc1-6b15bd833b1f":"Active military",
        "aed5eb81-1b7f-4040-9521-755f0ead2837":"Inactive military\/Veteran",
        "6c009e11-1d9c-4c7e-b67c-8cfcc9503580":"Temporarily unemployed",
        "dd7dc596-dd33-40ce-a9a5-98ce124f737b":"Full-time homemaker",
        "79379439-3abb-4c0a-8973-f152140fc9bc":"Retired",
        "817508d8-c0da-4019-9b61-3f1830932add":"Student",
        "b7902aac-bcbb-4e00-a790-23fe3c8f7a97":"Disabled",
        "6c0fe322-c622-48d5-baaa-7efc3a72de42":"Prefer not to answer",
        "d1f3cdd3-96e5-4046-983d-28f34775c02e":"Rent",
        "7b429355-dfc2-4ef9-be3b-a7d3336c8e5c":"Other",
        "d569741c-6004-429d-925c-185facfc3c4a":"Female teen age 14",
        "c78c255a-a6b1-4d03-b396-2ec8340a7ce5":"Male teen age 15",
        "cdc03e5c-fa36-409e-aeba-164156ff0d22":"Female teen age 15",
        "d85c79ff-ed85-4e07-8316-583971e17dcf":"Male teen age 16",
        "ccc9405c-7a3f-4215-b4a2-10259594c1d1":"Female teen age 16",
        "1e3574b4-fc2b-480a-bbec-394c948ab325":"Male teen age 17",
        "8103c4b9-6ba2-474a-a200-44699f1b2e0f":"Female teen age 17",
        "1fe42e99-e511-49e1-8e13-4b027488159b":"Amphibians (frogs, toads, etc.)",
        "0d759b88-a7ea-46b0-9565-a78b3630d120":"Bird(s)",
        "20c6c6e6-fb90-4073-81f1-1a8ee5d191a4":"Cat(s)",
        "ee0f951f-6c48-4661-9efd-f8c17f31a3cd":"Dog(s)",
        "2347fc09-397f-4baf-9239-51c50f309525":"Fish",
        "44932fdf-f31d-4099-ad3b-7378de29537c":"Horse(s)",
        "95620301-99cb-4683-930a-8605de11f3e3":"Reptiles (turtles, snakes, lizards, etc.)",
        "2961a507-d65c-4d41-9ab6-8734aac5a589":"Small animals or rodents (hamsters, mice, rabbits, ferrets, etc.)",
        "a7b0a522-cbb8-4252-abf7-51c9b912af1e":"Other",
        "a5e62039-56f4-429d-b6e1-573a5ac87d2a":"I do not have any pets",
        "e6f4ae10-5cb4-4de3-803a-9cd3ab78e952":"YOUNGSTOWN | 536",
        "9d0f4e52-7702-4a54-8bcf-0f912b0689cb":"YAKIMA-PASCO-RICHLAND-KENNWICK | 810",
        "0f2137bf-86aa-4303-a835-d28082294255":"Yes",
        "236b46c8-28e8-4b6c-8bf7-57e47713e4a1":"No",
        "5e68bab3-b4cf-483c-8248-233a2146ea8e":"Acne",
        "3dbca12a-37a2-4651-9258-f02122efc189":"ADD\/ADHD",
        "bd3c6d91-e2d4-446a-a011-a4530a95964a":"Allergies (not associated with Hay Fever)",
        "ee2ff007-f632-41ba-a5ca-aa324db898ba":"Alzheimer's",
        "3086c969-6326-44f9-af2c-ba8d87953058":"Anemia",
        "b4e4997f-e366-4089-b22d-dc73474f449a":"Angina",
        "669c2cdf-0bb6-4f2a-ad48-31324aefd71f":"1 person",
        "08b8654b-b5e4-4b32-b91d-88bfd48b5692":"2 persons",
        "531b0e03-a928-4f6e-a85e-4a738dac3394":"3 persons",
        "00578e4d-c3dd-4812-a9d8-310f9ca5a49a":"4 persons",
        "92af574f-90ee-4873-9688-3c5b674cf720":"5 persons",
        "643f9490-a269-483b-a5e1-0efac6c87e9e":"More than 5 persons",
        "d42b37b2-227f-4593-8939-04048d1efb55":"Prefer not to answer",
        "6663b80f-c344-447b-b0f7-5a927294aab3":"NEW HAMPSHIRE",
        "bfe2dabc-36ff-4cc3-8eb3-5e96a38fab47":"NEW JERSEY",
        "f1bc24e9-15d5-4040-bef3-d172dea89e28":"Anxiety",
        "79c526cc-5d7e-4e04-9a56-f4daf0287ad4":"Arrhythmia\/Atrial Fibrillation",
        "fdcbb5c9-8e37-492e-bd98-1852e79c32b7":"Arthritis",
        "0c9074e2-07ab-4353-bc3a-b1116f483e50":"Asthma",
        "aa257774-0f58-4703-9239-3d65b026db58":"Back Pain",
        "ffe02bf1-b154-45ab-b916-a0d9cedbb6d5":"Bipolar Disorder",
        "cecebe0c-43ed-424a-abb1-7ddc8b0e1b21":"Blood Disorders (non Cancerous)",
        "33ceb38d-9b79-4602-acbc-ef63356b181a":"Bronchitis",
        "4b88b5c7-4e73-48b7-86c4-086d340f9e79":"Cancer",
        "638e3385-f815-40a8-ba5a-40a707d7f692":"Cardiovascular Disease",
        "ef710996-ea6f-4850-8332-e649edc0f254":"Carpal Tunnel Syndrome",
        "6da7b9ba-bc3d-49a0-8ceb-1e47c10fb51c":"Chronic Fatigue Syndrome",
        "81fe61f6-5711-44f9-8d27-1e60354114d4":"Chronic Kidney Disease",
        "688112bf-2393-4c86-a04f-e938e09de358":"Chronic Lymphocytic Leukemia",
        "d19ed413-f798-4694-9131-7adcb205dda5":"Administration\/General Staff",
        "4c3d3e35-74c0-4828-957b-6de7cb783830":"Customer Service\/Client Service",
        "1228349b-2fb1-499e-b9cd-951a1187584d":"Executive Leadership",
        "21899bde-f394-4598-a139-c4bdc7815ad2":"Finance\/Accounting",
        "d67b7919-d6f6-4658-ad25-9d2463e97745":"Human Resources",
        "8cd8df79-c478-45a4-bdbe-34d2dddb361e":"Legal\/Law",
        "a8d85921-9489-4a84-ab09-4bc1360612de":"Marketing",
        "bd72a3ea-b6c8-43d8-a07c-78135b8d4bf0":"Operations",
        "e79279cf-473b-4e07-84a3-ab772d0231c3":"Less than $14,999",
        "90939ef1-8966-44af-b618-c855e0ab010e":"$15,000 to $19,999",
        "3126c47f-ad37-4ad5-b571-79b29acae0de":"$20,000 to $24,999",
        "bb530857-57eb-4749-84a2-9c00aad98567":"$25,000 to $29,999",
        "f2bb18aa-62fc-4bc9-91c0-cb70f745e5bd":"$30,000 to $34,999",
        "dc9bb9a8-9b6e-4cb0-befc-09fe93bee013":"$35,000 to $39,999",
        "8a2acfee-41f3-426c-abda-02feae77d570":"$40,000 to $44,999",
        "93e0095c-afdd-4d08-878f-c8cfaf527b99":"$45,000 to $49,999",
        "b1b2b217-6a0e-4557-95c0-a3490349ffb8":"Accounting",
        "931a124c-4c07-4caa-a352-a3ac5dee92be":"Advertising",
        "892c37fc-e6d2-4737-a23d-32efbc958516":"Agriculture\/Fishing",
        "baa2a08f-18ce-4f38-9f04-06291ed149bc":"Architecture",
        "b059bcbf-a045-4ef5-86b8-ee70888da2b1":"Automotive",
        "f7b902b8-209d-4eb4-ae6d-76043bd144e5":"Aviation",
        "809de6fa-8c1b-4070-ae56-3d8a1f434d93":"Banking\/Financial",
        "459688d6-441a-4e71-a95b-d4b533bc24f1":"Bio-Tech",
        "e931d8b4-9e06-4f8c-9dae-9b9d6d5625d5":"Brokerage",
        "64feba50-0f4f-4096-948e-06580aa1837a":"Carpenting\/Electrical installations\/VVS",
        "e090b5b5-dcc9-45f3-b1f6-cf3c8d16b4fe":"Chemicals\/Plastics\/Rubber",
        "f2a2f722-5826-4d50-bc3a-315d8181de62":"Communications\/Information",
        "99eabdd1-7aa2-4b12-9d41-4294052b15f3":"1",
        "1aeb0d8a-238e-48d9-8783-bb69598676c1":"2-10",
        "85ea98e9-5768-48bd-9371-42cfc8446301":"11-50",
        "5d94b806-98da-4aa4-8b1f-08091251c419":"51-100",
        "bebd8691-fae6-4bfb-9c7b-df41221d9a2a":"101-500",
        "f524175e-f2fa-4d30-b4ee-40183a2abae3":"501-1000",
        "b9227085-e9b5-499b-92ce-02603d41901d":"1001-5000",
        "2864d24b-bc60-4358-bc11-960668b287af":"Greater than 5000",
        "be8bf53f-9a02-48ee-8871-348058bebf66":"I don't work\/I don't know",
        "380fa529-836f-4237-8cf6-ca157768a728":"Computer Hardware",
        "3926255a-b8c1-4e31-af8d-3a2cf3992bc9":"Computer Reseller (software\/hardware)",
        "ff0f9696-9dbf-4e44-a998-60ce0c37e66a":"Computer Software",
        "c1db4b86-6d6b-499b-a5c0-ac39e7579d41":"Construction",
        "f43fdc98-0c6c-425e-9cee-bef173d317b0":"Consulting",
        "47f5d8b4-d493-4b4a-89bd-4e430bda2b4f":"Consumer Electronics",
        "cff7b90b-cf1c-40bf-a164-df8524008b90":"Consumer Packaged Goods",
        "cabf337b-9238-4533-afe5-7858e9e37882":"C-Level (e.g. CEO, CFO), Owner, Partner, President",
        "4f6af33b-4b43-427f-a3b9-da970939632f":"Yes",
        "08a0d0cc-33aa-4192-9cd9-3d143967357d":"No",
        "cdefdafc-ea38-44c0-ada8-1878dbe53a23":"Share decisions equally",
        "ab6328bc-4fce-4e23-86ba-181552f195ad":"Male",
        "88fcfb9e-45e7-4d89-9024-22e8fa4132b2":"Female",
        "cb905a21-cdff-4b74-887e-3399543a42af":"Age",
        "e91c9409-f705-4c44-a403-70baa1ac0eea":"None of the above",
        "e7b13bbf-5ea5-4cd8-a6d4-9deee90d98da":"Hepatitis",
        "e17829a8-f788-493f-8fa1-375c10b3c3f2":"High Cholesterol",
        "c7520586-868e-4470-8dc2-71ff2175c690":"HIV - Aids",
        "f30afb48-9bbe-49d4-8a7c-b12e6dbdc298":"Hypertension",
        "b534563b-54f6-4865-aeff-8296a43d7d22":"Hypothyroidism",
        "31636968-8da0-4e3e-81fa-43ad61433905":"Impotence",
        "6e318213-d9dd-476c-ae51-26fd52d062c5":"Infertility",
        "a86de9b0-f91f-403a-bc7a-6fc130eb450c":"Irritable Bowel Syndrome",
        "2c5f4cca-f3f6-4783-a84b-56c318902fc9":"Joint Replacement",
        "0e686f31-34ef-4d1e-8900-87577ce83ea3":"Kidney Failure",
        "52894e8e-59c2-4528-b342-2a81c0ea0f05":"Liver Cirrhosis",
        "3483a1de-b191-40e3-91e5-650842bc3212":"Liver Disorders (other than cancer or cirrhosis)",
        "39828565-0a1a-4c8b-8567-76751fa5c6da":"Lupus",
        "ea12d380-372f-470e-97e6-2ed76813ed6d":"Lyme Disease",
        "b192dacd-883c-4e16-992e-6074ea8319bb":"Menopause",
        "c4bf9b28-32e3-4be5-80ed-2fe36a3979c7":"Migraine",
        "5484e7ce-da23-4231-b878-9500ff2e352b":"Motor Neuron Disease",
        "64137fb0-7d9a-489e-99e5-f70ae4d59257":"Multiple Sclerosis",
        "53e4241e-9b2d-47b7-ae51-05e9b1435a7e":"Obesity",
        "f9643ead-407e-4527-b826-0e6d4401f0c0":"Osteoarthritis",
        "8b79fae1-9898-4911-9922-ac087b6808d2":"Osteoporosis",
        "f734fa7d-cee8-4cac-b922-17383800b489":"Overactive Bladder",
        "fa3a18d2-ec64-40e4-bf93-cc895828c9e8":"Parkinson's disease",
        "e6fe66fa-f07a-4009-b948-2d0fc2ae45dd":"Vice President (EVP, SVP, AVP, VP)",
        "9cb1cd3d-ab51-46d0-8990-b13d5c4f2ce7":"Director (Group Director, Sr. Director, Director)",
        "3aaf37b4-d065-48fb-8fa0-eaf10f2c4648":"Manager (Group Manager, Sr. Manager, Manager, Program Manager)",
        "21f96f8f-cb2e-4108-8011-2c58a6c982f0":"Analyst",
        "ed991885-dcb6-4661-903b-0feb53e8e441":"Intern",
        "1a3dbe74-13ab-4e01-b864-5330bca82769":"Pneumonia",
        "b5b37c7b-1ee8-46c6-a0dc-be0aa697f1c1":"ZIP",
        "f12bb586-e179-455f-81ee-0b27b88af81c":"British Columbia \/ Colombie-Britannique",
        "0bf34a92-8433-457f-8a77-0d94e7bccad7":"Manitoba",
        "8965093c-3b4c-438c-942a-6029f5a0a3ab":"New Brunswick \/ Nouveau-Brunswick",
        "6fa0cd26-54d5-4ef4-bd8a-99958b2264b4":"Nova Scotia \/ Nouvelle-?????cosse",
        "c96b22fb-da1a-450d-8da4-cba982bdd76b":"Ontario",
        "3cd360d1-152d-4d37-9fec-73a947348ee8":"Northwest Territories \/ Territoires-du-Nord-Ouest",
        "7be217df-eefb-4b7e-89f1-c5345e613ed5":"Prince Edward Island \/ ????le-du-Prince-?????douar",
        "7cc06e3a-5c51-4f56-a457-d205abe09f2f":"Qu????bec",
        "663443d9-822b-45f8-8cf3-55ae9eeb5e4e":"Yukon",
        "c1d73afa-83e9-45b7-97ff-91675053b871":"Saskatchewan",
        "d2e178d1-1fd1-47ed-95c3-e03e44f5c1c2":"Newfoundland and Labrador \/ Terre-Neuve-et-Labrador",
        "f2f7c0c1-3c0f-4234-98b7-f075f131e088":"Nunavut",
        "48308ef1-0691-44fb-8edb-a9fdb63a1d14":"Alberta",
        "42aa6b7f-4f94-44d2-84f8-20824d869e31":"Premature ejaculation",
        "aba42baf-4b77-4c48-8c83-6e7ecf1fc9fa":"Psoriasis",
        "ab0af8ba-1186-482b-a2ce-18ed7509c298":"Reflux",
        "7f9c236b-ff77-4aac-8d88-21b7c69b8233":"Restless Leg Syndrome",
        "1af8b6ab-0580-47eb-852d-1d5fb45cd844":"Rheumatoid Arthritis",
        "708a9b7b-2909-4970-9768-638b41d71810":"Schizophrenia",
        "7b68af03-55fc-461d-9bad-9da8e1d6d616":"Shingles",
        "39ecb993-ffa3-4886-bc9a-8ef8706d6abc":"Sinusitis",
        "0476d850-0e20-4f4a-8acb-23c55b12e200":"Sleeping Disorder",
        "a0992cbf-0687-4eab-91f3-ff0463a8cc18":"Smoking Addiction",
        "cc941916-72eb-42b8-9184-3865c9fbb34a":"Stroke",
        "09051179-7d9f-42e4-b9c6-3ddf66d997fc":"Substance Abuse Drugs\/Alcohol",
        "65682407-a9af-4372-b7d3-42ec9f3304c8":"Syphilis",
        "89d1b87a-ca39-4311-842e-5c7fe0376464":"Thyroid Problems",
        "88c1a366-6f08-4a7b-b585-21affb54f642":"Tuberculosis",
        "fed908f3-cec6-404c-84bc-e4378f6e8f53":"Ulcerative Colitis",
        "6a5abe91-26fa-473b-a9dd-f9002ac24945":"Ulcers",
        "3fe4859b-81e1-4c55-a03c-e48f99f79e51":"Urinary Incontinence",
        "922dba67-a3a8-47e7-b8c4-31518b829556":"I don't have any illnesses\/conditions",
        "b89218de-686e-4ca6-8048-6f50c61e9160":"Other",
        "9f008a33-9698-46e0-a496-246ece474b99":"Prefer not to answer",
        "c2cc66b9-cf0b-4b93-8833-8d828c828330":"DETROIT | 505",
        "13bebea7-9cd4-42ac-95c8-281ee3dab4c9":"DISTRICT OF COLUMBIA",
        "f85bfe20-50b5-40df-a28e-33b7a328d7a6":"FLORIDA",
        "5ad48eaa-54c1-4a3d-86b2-eff339b228e8":"GEORGIA",
        "e14cdb1d-c023-49a0-ac72-f7459edb015b":"HAWAII",
        "6459bb1e-098e-4038-b346-daef14ac0f4d":"IDAHO",
        "6d108ead-4e91-4129-b022-482a26b72431":"None of the above",
        "a32c1dc4-9f8d-48ab-9286-949b1ce1b190":"WILMINGTON | 550",
        "2b81223c-8d4f-4d99-a29b-5ce452dc16f5":"WHEELING-STEUBENVILLE | 554",
        "af6302bd-606c-4844-8e12-3af43cbb2750":"WILKES BARRE-SCRANTON | 577",
        "3775a8d1-0b34-4559-98d0-b9191c6a46ce":"ZANESVILLE | 596",
        "aafc1dde-f3c7-4d1d-a5e9-bb882f7efd62":"WICHITA FALLS & LAWTON | 627",
        "7aadcfc6-341f-4634-a0f6-ecf370694a97":"WICHITA-HUTCHINSON PLUS | 678",
        "aed308cb-ddd7-4707-b0ed-a8e6f720b051":"YUMA-EL CENTRO | 771",
        "f81715ab-fe4f-4f36-96bf-a2c5f8bcffd3":"ILLINOIS",
        "bd839603-cf9d-4dca-84f3-bf72b16e21c5":"INDIANA",
        "2ea429b7-0aaf-47cf-ae5a-4d5cad6f4fce":"IOWA",
        "2ec01ca3-9425-46ef-b0b6-3df66183f4c7":"NEVADA",
        "58b4a531-fd00-49c8-80a1-2a8d320d09a2":"ALABAMA",
        "2f201a34-b2e2-4e87-8b93-5fcc079bd5d0":"ALASKA",
        "95180995-3650-4f8a-9615-f3c0b9585d8e":"ARIZONA",
        "957ca375-7bdb-4515-ab58-d8712762ec93":"ARKANSAS",
        "576d0379-88d9-4498-9122-a934a7aea4ba":"CALIFORNIA",
        "c7a8a435-9106-47be-8212-3f234014cffa":"COLORADO",
        "6161ca39-7d36-414c-82a7-176c7ac225a0":"CONNECTICUT",
        "7ec51e52-1778-4be5-b0d3-1d5fa54d2be6":"DELAWARE",
        "1c841162-5fab-44de-9128-308bd563bac9":"KENTUCKY",
        "248adb47-9f53-4b3c-8571-73079540d442":"LOUISIANA",
        "f7211023-7d7e-43c6-be05-d073dd7cb08c":"MAINE",
        "c29a9c99-4932-4f40-8b20-3f0a57d66777":"Federated States of Micronesia",
        "78c0afb9-260c-4ba8-b86d-84cb7f5d95e1":"Guam",
        "040771bc-b9b2-4de0-837d-0f37da745193":"American Samoa",
        "35000a0e-8d4e-4a84-b684-6be0fb22cc17":"Marshall Islands",
        "df2b4aba-c89d-47be-b5d7-92649829ba85":"MASSACHUSETTS",
        "ce60bf3b-5967-4865-89bb-7005041a4fe4":"MICHIGAN",
        "e71697f9-e341-4eeb-a316-cdb6b739a472":"MINNESOTA",
        "d67ea0c7-c275-490b-8e6a-28c3f828f661":"MISSISSIPPI",
        "5a03d5fc-8b03-4f94-af5e-64f1941b9321":"MISSOURI",
        "e4501774-3be7-4b4b-9ddf-6288346c0551":"MONTANA",
        "7a6942b0-defa-4f6f-b38a-21489d114801":"NEBRASKA",
        "c7f58d8e-9ea3-4185-82b3-b962446b5728":"NEW MEXICO",
        "3a3bb518-fe10-41f0-a463-bf921736aebe":"Midwest",
        "a2bf30f4-94c4-474d-955e-94db3f16254f":"Northeast",
        "99a8380d-9669-459c-b1b7-62ceaff63a0f":"South",
        "80d64d1e-52e3-4a08-a3a2-b5e1883d11bc":"West",
        "43b0647e-d32b-47f2-9e43-4c2b1fd8c758":"NEW YORK",
        "1bdf6272-2b21-4bbe-bc2d-364a4560b456":"NORTH CAROLINA",
        "80637da1-ecba-4212-ac5a-62efb1979e26":"NORTH DAKOTA",
        "24f9e516-03da-4ccf-bd0c-d01b63c762a5":"OHIO",
        "4362dcc3-dbc7-4ffb-ac52-a97cf480b26e":"OKLAHOMA",
        "641c2cfc-7857-4c17-882b-adef4c3a4f75":"OREGON",
        "37470563-4af2-4658-8f58-9577df6d93ef":"PENNSYLVANIA",
        "a29f6574-6552-4187-9ecb-aab96756230c":"RHODE ISLAND",
        "9770efb7-e6a0-4d82-b7b3-d2843c45b25b":"SOUTH CAROLINA",
        "398de2e2-e822-4653-8e39-404d33ef3f4e":"SOUTH DAKOTA",
        "65acf78d-cb4e-4c4f-886b-0f0e58bdf8a2":"TENNESSEE",
        "6bfccc55-4292-49d1-aec6-b6db7606f510":"TEXAS",
        "1b7ecb6c-3fb5-4fa4-b047-b3e2c97c6346":"UTAH",
        "84c9cb7d-4ff2-480d-a0d0-152070545d97":"VERMONT",
        "e49357ef-09f7-445d-abea-fef2cb27fed1":"VIRGINIA",
        "afdc43e9-e42b-45dd-8654-52de3757e1ce":"WASHINGTON",
    }

    r = []
    for i in lst:
        ret = []
        for j in i['values']:
            if d.get(j):
                ret.append(d[j])
            else:
                ret.append(j)
        
        r.append({c[i['id']]: ret})

    # print(r)
    return r

def study_groups():
    global df

    print(f'[{cn}]: Evaluating datatype (list)')
    df[cn] = df[cn].apply(ast.literal_eval)

    print(f'[{cn}]: Exploding into multiple rows')
    df = df.explode(cn)

    print(f'[projects__study_types_ids]: Creating and mapping ...')
    df['projects__study_types_ids'] = df.apply(lambda x: map_study_id(x[cn]['id']), axis=1)

    print(f'[projects__study_types_subject_ids]: Creating and mapping ...')
    df['projects__study_types_subject_ids'] = df.apply(lambda x: map_subject_id(x[cn]['studySubjectIds']), axis=1)

    df = df.explode('projects__study_types_subject_ids')


def split_df_subject_ids():
    global df

    print('[df]: Splitting by projects__study_types_subject_ids')
    splits = list(df.groupby('projects__study_types_subject_ids'))

    for s in splits:
        del s[1]['projects__study_types_subject_ids']

    return splits

def split_df_continents():
    global df

    print(f'Splitting by continent...')
    splits = list(df.groupby('continent'))

    for s in splits:
        del s[1]['continent']

    return splits

def target_groups(df):    
    dn = 'projects__target_groups'

    print(f'[{dn}]: Evaluating datatype (list)')
    df[dn] = df[dn].apply(ast.literal_eval)

    print(f'[{dn}]: Exploding into multiple rows')
    df = df.explode(dn)

    print(f'[projects__target_groups_name]: Creating ...')
    df['projects__target_groups_name'] = df.apply(lambda x: x[dn]['name'], axis=1)

    print(f'[projects__target_groups_quota]: Creating ...')
    df['projects__target_groups_quota'] = df.apply(lambda x: x[dn]['quota'], axis=1)

    print(f'[projects__target_groups_qualifications]: Creating and mapping ids...')
    df['projects__target_groups_qualifications'] = df.apply(lambda x: map_qual(x[dn]['qualifications']), axis=1)

    print(f'[projects__target_groups_qualifications_combine]: Creating ...')
    df['projects__target_groups_qualifications_combine'] = df.apply(lambda x: dict(ChainMap(*x['projects__target_groups_qualifications'])), axis=1)
    
    return df

def split_quals(df):
    print(f'[df2]: Creating new dataframe from qualifications_combine column')
    df2 = pd.json_normalize(df['projects__target_groups_qualifications_combine'].tolist())

    df2.columns = [f'qualifications_{i}' for i in df2.columns]

    print(f'[df3]: Concatenating: df + df2 on axis=1')
    df = pd.concat([df.reset_index(drop=True), df2], axis=1)

    return df

def save_to_csv_continent(df, output_file):
    filename = f'information_retrieval/data/dfs/continent_{output_file}.csv'

    print(f'\n[FILE]: Exporting DataFrame to {filename} ... \n\n')
    df.to_csv(filename, index=False)

def save_to_csv(df, is_train, output_file):
    #filename = f'data/output/{output_file}.csv'
    if is_train==True:
        filename = f'information_retrieval/data/{output_file}.csv'
        print(f'\n[FILE]: Exporting DataFrame to {filename} ... \n\n')
        df.to_csv(filename, index=False)
    else:
        filename = f'information_retrieval/data/inference_transformed_data.csv'
        print(f'\n[FILE]: Exporting DataFrame to {filename} ... \n\n')
        df.to_csv(filename, index=False)        

def main_transform(data_transform_path, is_train):

    global df
    global input_file
    # input_file = data_transform_path
    # df = pd.read_csv(input_file)
    df = data_transform_path

    dfs = []
    if is_train == True:
        unique_filter()
    map_continent()
    study_groups()
    #combine_lang_study()
    combine_continent_lang_study()
    #dfs = split_df_subject_ids()
    #dfs = split_df_continents()

    print('\nTotal Dataframes after splitting: ', len(dfs))
    
    if len(dfs) != 0:
        for name, df in dfs:
            df1 = target_groups(df)
            df2 = split_quals(df1)
            return df2
            # save_to_csv_continent(df2, name)
            # '''
            # df1_qualification_subset = df1[['sample_pulls__language','projects__study_types_ids','projects__target_groups_qualifications_combine']]
            # #df1_qualification_subset.reset_index(inplace=True)
            # df1_qualification_subset = df1_qualification_subset.reset_index(drop=True)
            # df1_qualification_subset.to_json(f'continent_jsons/continent_{name}.json', orient = 'columns')
            # '''
        
    else:
        df1 = target_groups(df)
        df2 = split_quals(df1)
        return df2
        # save_to_csv(df2, is_train, 'final_transformed_data')
        # '''
        # df1_qualification_subset = df1[['sample_pulls__language','projects__study_types_ids','projects__target_groups_qualifications_combine']]
        # #df1_qualification_subset.reset_index(inplace=True)
        # df1_qualification_subset = df1_qualification_subset.reset_index(drop=True)
        # df1_qualification_subset.to_json(f'continent_jsons/continent_{name}.json', orient = 'columns')
        # '''
        
    print('\n[SUCCESS]: Task completed!')
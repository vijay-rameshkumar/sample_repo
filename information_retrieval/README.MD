<h1> Information Retrieval</h1>
- This is the First Component of the Supplier Recommendation Engine.
<br>- The task of this component is to match the requirements of the QM project with the probable suppliers who might have supplied panelist in the past for the given combination of the requirement.
<br>- This is done so as to get the list of suppliers and pass it to the next component i.e. Recommender System.

<h2> Data Transformation :</h2>
<img src="output/data_transformation_flow.png" width="50%" height="50%">
- The above transformations are done on the source file i.e. Panelist-wise Disposition csv.
<br>

<h2> Model Flow </h2>
<img src="output/Model_flow.png" width="50%" height="50%">
- There are Two IR models used as shown in the flow above.
<br>

<h2> Target Group Qualification Transformation </h2>
<img src="output/target_group_qualification_transformed_tree_snapshot.png" width="50%" height="50%">
- Above is the depiction of how the Target Group Qualification column is transformed and flattened as it is unstructured and dynamic in nature.
<br>

<h2> Target Group Qualification Sample </h2>
<img src="output/target_group_flattened_csv_snapshot.png" width="50%" height="50%">
<br>

<h2> IR Model Validation Sample Sheet </h2>
<img src="IR_model_validation_sheet.png" width="100%" height="50%">
<br>

<h1> Training Guide :</h1>
python main.py

<h1> Inference Guide :</h1>
python inference.py

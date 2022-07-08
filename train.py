#import libraries
from multiprocessing import Process
import recommender_system.main as recsys_train
import information_retrieval.QM_IR_train_script as ir_train
import data_transformation as dt
# parallel trainings
# def ir_train():
#     print("IR training")

# my_process1 = Process(target=ir_train, )
# my_process2 = Process(target=recsys_train.train(),)

# my_process1.start()
# my_process2.start()

# my_process1.join()
# my_process2.join()
data_transform_path = 'data/source/source_data.csv'
is_train = True
dt.main_transform(data_transform_path, is_train)
ir_train.train()

recsys_train.train()
print ("Training Completed")
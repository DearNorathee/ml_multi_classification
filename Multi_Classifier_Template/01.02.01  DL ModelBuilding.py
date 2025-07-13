#%%
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pyarrow
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import preprocessing


######################################################### Change Input Below ################################################
filename = r"C:\Users\n1603499\OneDrive - Liberty Mutual\Documents\006.01  Data Science Challenge(DSC 2022)\sprint_real_dataset.parquet"

my_epoc = 100
feature_cat_name = ["Loss_Type", "Damage_Type","Lot_Run_Condition","Lot_Make"]
y_name = "Sale_Price"
#Adjust Learning rate
myOptimizer = Adam(learning_rate = 0.1)
# Activation functions to try out
activations = ['relu','tanh','sigmoid']
seed = 10
my_funct = 'relu'
n_layer = 4
n_node = 20
loss = 'mse'
optimizer = 'adam'
n_data = 10000

#%%
######################################################### Change Input Above ################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Define the model @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def make_NN(X,y):
    global scaler_x, scaler_y
    scaler_x = MinMaxScaler( feature_range=(0,1))
    scaler_y = MinMaxScaler( feature_range=(0,1))

    scaled_X = scaler_x.fit_transform(X)
    y = pd.DataFrame(y)
    scaled_y = scaler_y.fit_transform(y)

    scaler_x.fit(X)
    scaler_y.fit(y)

    m = scaler_y.scale_[0]
    k = scaler_y.min_[0]
    print("Note: y values were scaled by multiplying by {:.10f} and adding {:.6f}".format(m, k))
    input_nCol = len(X.axes[1])
    n_data = len(X.axes[0])
    model = MakeModelDL(input_shape = input_nCol,n_layers= n_layer,n_nodes=n_node,act_function= my_funct,task_type="regressionPlus")
    model.compile(loss= loss,optimizer=optimizer)
    model.fit(scaled_X,scaled_y,epochs=my_epoc,shuffle=True,verbose =0)
    return model
def NN_Predict(model,x_new):
    scaled_x = scaler_x.transform(x_new)
    predict_val = model.predict(scaled_x)
    predict_val = scaler_y.inverse_transform(predict_val)
    return predict_val
    
    


#%% prepare file
filepath = r"H:\W_Documents\DSC 2022\sprint_real_dataset.parquet"
parquet_file = pd.read_parquet(filepath)
# replace 'na' and 'nan' with np.nan
parquet_file = parquet_file.replace('na', np.nan)
parquet_file = parquet_file.replace('nan', np.nan)
# parquet_file.info(verbose = True, null_counts=True)
# remove date data out of numeric df
date_columns = ['Auction_DT_week', 'Auction_DT_month','Auction_DT_year']
not_use_col = ['Auction_DT_week', 'Auction_DT_month','Auction_DT_year','ElectricIndicator_HLDI','odomread','acv_pred_price',"RedesignYear",'Odometer_reading', "additional_collision_points_desc" , "claim_open_dt" , "claim_rptd_dt" , "CVRG_TYPE_CD" , "external_source_type" , "init_point_of_impct_type" , "loss_st_cd" , "policy_st_cd" , "text_description" , "vehicl_color_nme" , "vehicl_towed_type" , "Copart_Facility_Name" , "FNOL_DT" , "Keys" , "Loss_DT" , "Pickup_DT" , "Sale_Title_Type" , "BasePrice_DDM4_Desc" , "BodyStyle"]
parquet_file = parquet_file.drop(not_use_col,axis=1)
parquet_file = parquet_file[parquet_file['Lot_ACV'] != 0]
#--------------------------------------------------------------------------------------------------------
# import data and set lot id as index (use id to join later if we did seperate)
#--------------------------------------------------------------------------------------------------------
## checking for data types - can check this too with info() method
dtype=parquet_file.dtypes
cate_var = []
num_var=[]
ot_var=[]
mySeed = 12
n = 20
for f, t in dtype.items():
    if t==object:
        cate_var.append(f)
    elif t==float:
        num_var.append(f)
    else:
        ot_var.append(f)     
cate_used_var=["Damage_Type","Loss_Type","Lot_Make","Lot_Model"]
data_cate=parquet_file[cate_used_var]
data_num=parquet_file[num_var]
fulldata=pd.concat([data_cate,data_num],axis=1)

fulldata = fulldata[0:n_data]

#%%
#label encoding
def create_label_encoding_with_min_count(fulldata, column, min_count=0.01*fulldata.shape[0]):
    column_counts = fulldata.groupby([column])[column].transform("count")
    column_values = np.where(column_counts >= min_count, fulldata[column], "")
    fulldata[column+"_label"] = preprocessing.LabelEncoder().fit_transform(column_values)
    
    return fulldata[column+"_label"]
for cat in cate_used_var:
    fulldata[cat]=create_label_encoding_with_min_count(fulldata, cat)
    
#X = data.drop(['Sale_Price','Loss_Type','Damage_Type', 'Lot_Run_Condition', 'Lot_Make'],axis=1)
#Y = data['Sale_Price']
X = fulldata.drop(['Sale_Price','Lot_ACV','Damage_Type', 'Loss_Type', 'Lot_Make', 'Lot_Model'],axis=1)

parquet_file = pd.get_dummies(parquet_file, columns =feature_cat_name )
parquet_file = parquet_file[parquet_file['Lot_ACV'] != 0]
for f, t in dtype.items():
    if t==object:
        cate_var.append(f)
    elif t==float:
        num_var.append(f)
    else:
        ot_var.append(f)


#%% split the data validation and holdout


X_train, X_test = train_test_split(X,train_size=0.7,random_state=mySeed)
y_train, y_test = train_test_split(fulldata['Sale_Price'],train_size=0.7,random_state=mySeed)

sale_price_train, sale_price_test = train_test_split(fulldata['Sale_Price'],train_size=0.7,random_state=mySeed)
sale_price_train.reset_index(drop = True,inplace = True)
sale_price_test.reset_index(drop = True,inplace = True)
lot_ACV_train, lot_ACV_test= train_test_split(fulldata['Lot_ACV'],train_size=0.7,random_state=mySeed)
lot_ACV_train.reset_index(drop = True,inplace = True)
lot_ACV_test.reset_index(drop = True,inplace = True)

y_train = sale_price_train/lot_ACV_train
lot_ACV_train = np.array(lot_ACV_train)
lot_ACV_test = np.array(lot_ACV_test)



scale_train = MinMaxScaler( feature_range=(0,1))
scale_valid = MinMaxScaler( feature_range=(0,1))
scale_holdout = MinMaxScaler( feature_range=(0,1))

# scale_y = MinMaxScaler(feature_range=(0,1))
# X_train = train_df.loc[:,feature_cat_name]
# X_valid = valid_df.loc[:,feature_cat_name]
# X_holdout = holdout_df.loc[:,feature_cat_name]
# X_train =  train_df.drop([y_name],axis=1)
# X_valid = valid_df.drop([y_name],axis=1)
# X_holdout = holdout_df.drop([y_name],axis=1)

# y_train = train_df[[y_name]]
# y_valid = valid_df[[y_name]]
# y_holdout = holdout_df[[y_name]]

y_valid = y_test
X_valid = X_test

# scaled_X_train = scale_train.fit_transform(X_train)
# scaled_X_valid = scale_valid.fit_transform(X_valid)
# # scaled_X_holdout = scale_train.fit_transform(X_holdout)

# # scaled_y_holdout = scale_train.fit_transform(y_holdout)
# y_valid = pd.DataFrame(y_valid)
# y_train = pd.DataFrame(y_train)
# scaled_y_valid = scale_valid.fit_transform(y_valid)
# scaled_y_train = scale_train.fit_transform(y_train)

# print(scaled_X_train[0:10])

#%%
def MakeModelDL(input_shape,n_layers,n_nodes,act_function = 'relu',task_type = 'regression'):
    model = Sequential()
    model.add(Dense(n_nodes, input_shape = (input_shape,),activation= act_function ) )
    for i in range(n_layers-1):
        model.add(Dense(n_nodes, input_shape = (input_shape,),activation= act_function ))

    if task_type in ["regression","rg"]:
        model.add(Dense(1,activation = 'linear'))
    elif task_type in ["classify","classification"]:
        pass
    elif task_type in ["regressionPlus"]:
        model.add(Dense(1,activation = 'sigmoid'))
    # if task_type in ["regression","rg"]:
    #     model.compile(loss='mse',optimizer = 'adam')
    return model

def MakeModel_Nlayers():
    pass
def MakeModel_Nnodes():
    pass



#%%
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Define the model @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
model01 = make_NN(X_train,y_train)
train_pred = NN_Predict(model01,X_train)
train_pred = train_pred.flatten()
train_pred = train_pred * lot_ACV_train
valid_pred = NN_Predict(model01,X_valid)
valid_pred = valid_pred.flatten()
valid_pred = valid_pred * lot_ACV_test

y_train = np.array(y_train)
y_valid = np.array(y_valid)

rsme_train = np.sqrt(mean_squared_error(sale_price_train,train_pred))
rsme_valid = np.sqrt(mean_squared_error(sale_price_test,valid_pred))


# y_train = y_train.values.tolist()
# y_valid = y_valid.values.tolist()
y_compare_train = [sale_price_train,train_pred]
y_compare_valid = [sale_price_test,valid_pred]
for i in range(100):
    print(sale_price_train[i],train_pred[i])
print("*"*30)
for i in range(100):
    print(sale_price_test[i],valid_pred[i] )

print(rsme_train,rsme_valid)




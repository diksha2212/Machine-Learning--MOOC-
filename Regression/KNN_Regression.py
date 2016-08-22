## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

###LOADING DATA

housesmall_train=pd.read_csv("kc_house_data_small_train.csv")
housesmall_test=pd.read_csv("kc_house_data_small_test.csv")
housesmall_validation=pd.read_csv("kc_house_data_validation.csv")

feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
                
                
  ## get feature Matrix
def get_feature_matrix(data,features,output):
    data['constant']=1
    features=['constant']+features
    features_data=data[features]
    feature_matrix=np.array(features_data)
    output_array=np.array(data[output])
    return (feature_matrix,output_array)

##NORMALIZE FEATURES

def normalize_features(features):
    X=features
    norms = np.linalg.norm(X, axis=0)
    normalized_features=X / norms
    return (normalized_features, norms)
    
##GETTING FEATURE MATRIX FOR DIFFERENT DATA
features_train, output_train = get_feature_matrix(housesmall_train, feature_list, 'price')
features_test, output_test = get_feature_matrix(housesmall_test, feature_list,'price')
features_valid, output_valid = get_feature_matrix(housesmall_validation, feature_list,'price')

##NORMALIZING FEATURES

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

##chooSing query house
house_query=features_test[0]
house_query

##CHOOSING TRAIN HOUSE

train_house=features_train[9]

##COMPUTE A SINGLE DISTANCE

def calculate_distance(target_house,query_house):
  sum=0
  for i in range(0,17) :
   sum= sum+ ((target_house[i] - query_house[i])**2)
  distance=np.sqrt(sum)
  return distance
  
calculated_distance=calculate_distance(train_house,house_query)
calculated_distance

    
  ## calcualating k nearest distance ##
def calculate_distance(k,features_train,query_house):
  diff = features_train[0:len(features_train)]-query_house    
  distances=np.sqrt(np.sum(diff**2,axis=1))
  k_indices = np.argsort(distances)[0:k]
  return k_indices
  
  ##GETTING K NEAREST HOUSES
  calculate_distance(10,features_train,features_test[2])

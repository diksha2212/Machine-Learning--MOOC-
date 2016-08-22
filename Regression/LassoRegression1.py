## IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##LOADING DATA

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
              'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int,
              'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)

##FEATURE SELECTION

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']


##CREATING MODEL WITH ALL FEATURES
from sklearn import linear_model  # using scikit-learn

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True) # set parameters
lasso_model=model_all.fit(sales[all_features], sales['price']) # learn weights

##features chosen by lasso
lasso_model.coef_

##To find a good L1 penalty, we will explore multiple values using a validation set.##

testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

##FEATURE SELECTION FRO DIFFERENT DATASETS
testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

## fitting training data for various L1 penalty

for l1_penalty in np.logspace(1, 7, num=13):
    model_all = linear_model.Lasso(alpha=l1_penalty, normalize=True) # set parameters
    model_all.fit(training[all_features], training['price']) # learn weights
    print l1_penalty
    print get_residual_sum_of_squares(model_all,validation,validation['price'])
    
###LIMIT THE NUMBER OF NONZERO WEIGHTS

max_nonzeros = 7

##define a wide range of possible l1_penalty_values:

l1_penalty_values = np.logspace(8, 10, num=20)

for l1_penalty in l1_penalty_values:
  model_all = linear_model.Lasso(alpha=l1_penalty, normalize=True) # set parameters
  model_all.fit(training[all_features], training['price']) # learn weights
  print l1_penalty
  print model['coefficients']['value'].nnz()
  
  ##from this we get a min and max range of l1 penalties

## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

##Loading Data
house_train=pd.read_csv("kc_house_train_data.csv")
house_test=pd.read_csv("kc_house_test_data.csv")

example_features = ['sqft_living', 'bedrooms', 'bathrooms']

feature_train=np.array(house_train[example_features])
target_train=np.array(house_train['price'])
feature_test=np.array(house_test[example_features])
target_test=np.array(house_test['price'])

##SELECTING FEATURES AND APPLYING MODEL ##

from sklearn import linear_model
clf = linear_model.LinearRegression()
multiple_model=clf.fit(feature_train,target_train)

## getting coefficients
multiple_model.coef_

##Making Predictions ##

predictions = multiple_model.predict(feature_test)

###CREATING NEW FEATURES ###

from math import log

house_train['bedrooms_squared'] = house_train['bedrooms'].apply(lambda x: x**2)
house_test['bedrooms_squared'] = house_test['bedrooms'].apply(lambda x: x**2)

house_train['bed_bath_rooms'] = house_train['bedrooms']*house_train['bathrooms']
house_test['bed_bath_rooms'] = house_test['bedrooms']*house_test['bathrooms']

house_train['log_sqft_living'] = house_train['sqft_living'].apply(lambda x: log(x))
house_test['log_sqft_living'] = house_test['sqft_living'].apply(lambda x: log(x))

house_train['lat_plus_long'] = house_train['lat']+house_train['long']
house_test['lat_plus_long'] = house_test['lat']+house_test['long']

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

### getting data ready for different models

model1_feature_train=np.array(house_train[model_1_features])
model1__target_train=np.array(house_train['price'])
model1__feature_test=np.array(house_test[model_1_features])
model1__target_test=np.array(house_test['price'])

model2_feature_train=np.array(house_train[model_2_features])
model2__target_train=np.array(house_train['price'])
model2__feature_test=np.array(house_test[model_2_features])
model2__target_test=np.array(house_test['price'])

model3_feature_train=np.array(house_train[model_3_features])
model3__target_train=np.array(house_train['price'])
model3__feature_test=np.array(house_test[model_3_features])
model3__target_test=np.array(house_test['price'])

## CREATING MODELS
clf1=linear_model.LinearRegression()
model1=clf1.fit(model1_feature_train,model1__target_train)

clf2=linear_model.LinearRegression()
model2=clf2.fit(model2_feature_train,model2__target_train)

clf3=linear_model.LinearRegression()
model3=clf3.fit(model3_feature_train,model3__target_train)

## Getting Coefficients for each model

model1.coef_
model2.coef_
model3.coef_

###getting RSS for various data

def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)
    
    # Then compute the residuals/errors
    residuals= outcome - predictions
    # Then square and add them up
    residuals_squared=residuals*residuals
    RSS =residuals_squared.sum()
    return(RSS) 
    
    ### GETTING RSS ON THE TEST DATA 
get_residual_sum_of_squares(model1,model1__feature_test,model1__target_test)
get_residual_sum_of_squares(model2,model2__feature_test,model2__target_test)
get_residual_sum_of_squares(model3,model3__feature_test,model3__target_test)

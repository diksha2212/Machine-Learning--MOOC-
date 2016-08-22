## Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

##Loading Data
house_train=pd.read_csv("kc_house_train_data.csv")
house_test=pd.read_csv("kc_house_test_data.csv")

##setting features and targets

feature_train=np.array(house_train[example_features])
target_train=np.array(house_train['price'])
feature_test=np.array(house_test[example_features])
target_test=np.array(house_test['price'])

## get feature Matrix
def get_feature_matrix(data,features,output):
    data['constant']=1
    features=['constant']+features
    features_data=data[features]
    feature_matrix=np.array(features_data)
    output_array=np.array(data[output])
    return (feature_matrix,output_array)

##getting features for train data
(example_features, example_output) = get_feature_matrix(house_train, ['sqft_living'], 'price')
### test features and output
(test_features,test_output) = get_feature_matrix(house_test, ['sqft_living'], 'price')

##predicting output
def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
#helper method
def feature_derivative(errors, feature):
    derivative=2*(predict_output(feature,errors))
    return(derivative)
    
## GRADIENT DESCENT 
from math import sqrt
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged=False
    weights = np.array(initial_weights)
    while not converged:
        predictions=predict_output(feature_matrix,weights)
        errors=predictions-output
        gradient_sum_squares = 0
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]          
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:,i])
            # add the squared derivative to the gradient magnitude
            derivative_squared =derivative*derivative
            gradient_sum_squares+=derivative_squared
            # update the weight based on step size and derivative:
            weights[i]=weights[i]-(step_size*derivative)
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

### RUNNING GRADIENT DESCENT ###

initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

train_weights = regression_gradient_descent(example_features, example_output,initial_weights, step_size,  tolerance)
train_weights

test_weights = regression_gradient_descent(test_features, test_output,train_weights, step_size ,tolerance)
test_weights

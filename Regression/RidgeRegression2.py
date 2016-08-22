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

###GETTING FEATURE DERIVATIVE FOR RIDGE

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if(feature_is_constant):
      derivative = 2*(predict_output(feature,errors)) 
    
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
      derivative = 2*(predict_output(feature,errors)) + 2*l2_penalty*weight
    return derivative
    

(example_features, example_output) = get_feature_matrix(house_train, ['sqft_living'], 'price') 
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors

### GRADIENT DESCENT FOR RIDGE REGRESSION

from math import sqrt
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
    
    #while not reached maximum number of iterations:
    for j in range (1,max_iterations): 
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions=predict_output(feature_matrix,weights)
        # compute the errors as predictions - output
        errors=predictions-output
        gradient_sum_squares = 0
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            if(i==0):
             derivative=feature_derivative_ridge(errors, feature_matrix[:, i],weights[i], l2_penalty, True)
            #(Remember: when i=0, you are computing the derivative of the constant!)
            else:
              derivative=feature_derivative_ridge(errors, feature_matrix[:, i],weights[i], l2_penalty, False)
                
            derivative_squared=derivative*derivative
            gradient_sum_squares=gradient_sum_squares+derivative_squared
            # update the weight based on step size and derivative:
            weights[i]=weights[i]-(step_size*derivative)
                   # subtract the step size times the derivative from the current weight
            
    return weights
    
  ## VISUALIZING EFFECT OF L2 PENALTY ##

simple_features = ['sqft_living']
my_output = 'price'

(simple_feature_matrix, output) = get_feature_matrix(house_train, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_feature_matrix(house_test, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, 0.0, max_iterations)
simple_weights_0_penalty

simple_weights_high_penalty=ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size,1e11, max_iterations)
simple_weights_high_penalty

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
        
        
##RIDGE REGRESIION ON TEST DATA FOR VARIOUS WEIGHTS
test_weights=ridge_regression_gradient_descent(simple_test_feature_matrix,test_output,initial_weights ,step_size,0.0, max_iterations)
test_RSS = ((test_output-predict_output(simple_test_feature_matrix,test_weights))**2).sum()
test_RSS

test_weights=ridge_regression_gradient_descent(simple_test_feature_matrix,test_output,simple_weights_0_penalty ,step_size,0.0, max_iterations)
test_RSS = ((test_output-predict_output(simple_test_feature_matrix,test_weights))**2).sum()
test_RSS

test_weights=ridge_regression_gradient_descent(simple_test_feature_matrix,test_output,simple_weights_high_penalty ,step_size,0.0, max_iterations)
test_RSS = ((test_output-predict_output(simple_test_feature_matrix,test_weights))**2).sum()
test_RSS

##VISUALIZING
plt.plot(simple_test_feature_matrix,test_output,'k.',
         simple_test_feature_matrix,predict_output(simple_test_feature_matrix, simple_weights_0_penalty),'b-',
        simple_test_feature_matrix,predict_output(simple_test_feature_matrix, simple_weights_high_penalty),'r-')

### importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

##loading house sales data
house_train=pd.read_csv("E:\ML-data\Regression\kc_house_train_data.csv\kc_house_train_data.csv")
house_test=pd.read_csv("E:\ML-data\Regression\kc_house_test_data.csv\kc_house_test_data.csv")

### setting features and targets
feature_train=np.array(house_train['sqft_living'])
target_train=np.array(house_train['price'])
feature_test=np.array(house_test['sqft_living'])
target_test=np.array(house_test['price'])

## defining simple linear regression function
def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    total= input_feature+output
    
    product =input_feature*output
    
    sum_product=sum(product)
    feature_squared = input_feature*input_feature
    sum_feature_squared=sum(feature_squared)
    num=sum_product - ((sum(output)*sum(input_feature))/len(input_feature))
    den=sum_feature_squared - (sum(input_feature)*sum(input_feature)/len(input_feature))
    slope = num/den
    intercept = output.mean()-slope*input_feature.mean()
 
    return (intercept, slope)
    
    ## running linear regression for train data
    sqft_intercept,sqft_slope= simple_linear_regression(feature_train,target_train)
    
    ## Predicting values
def get_regression_predictions(input_feature,intercept,slope):
    predicted_values = (intercept + (slope*input_feature))
    
    return predicted_values

## predicting price for 2650 sqft
my_house_sqft = 2650
estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)

## Residual Sum Of Squares
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    
    residuals = output -get_regression_predictions(input_feature,intercept,slope)

    residuals_squared=residuals*residuals
    RSS =residuals_squared.sum()

    return(RSS)
    
    ##computing RSS
    print get_residual_sum_of_squares(feature_train,target_train,sqft_intercept,sqft_slope)
